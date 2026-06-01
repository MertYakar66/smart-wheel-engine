---
id: S47
title: Live wheel session 2026-03-20 trust audit on an elevated-vol tape
kind: usage
status: complete
terminal: ultracode
pr:
decisions: []
date: 2026-05-31
headline: Sat down and *used* the engine for a full wheel session at as_of=2026-03-20 (VIX 28.97, HMM bear). Verdict — TRUST IT FOR ENTRY (gating, strike/premium math, sizing-down, EV refusals all sound and realistic), DISTRUST IT FOR MANAGEMENT (suggest_rolls and the covered-call ranker go silent on challenged/assigned names by default — credit-only filter + basis-unaware strike grid). Probabilities are honest but coarse (35-DTE prob_profit = k/35 empirical counts, ±~6pp). Premiums are conservative (no put skew → ~12–20% under a real chain).
surface:
  - scripts/s47_trader_session_2026_03_20.py
---

## The headline question

> *When I bring this engine a real trading situation, does it give me reliable,
> realistic answers I could act on with real money?*

**Yes for the entry decision; no for the management decision.** Concretely:

| Phase | Trust? | Why (this session's evidence) |
|---|---|---|
| **What to sell today** (rank, gate, size) | **TRUST** | Strikes solve to the right delta (−0.25 ± 0.01); premiums are real BSM within <1–5%; the earnings gate, the negative-EV refusals, and R11's elevated-vol size-down all behave like a disciplined desk. |
| **Probabilities** | **TRUST THE BAND, NOT THE DIGITS** | `prob_profit` is an honest non-parametric count, but over only **35 windows** at 35-DTE → granularity 2.9pp, ±~6pp sampling error. Read 0.8857 as "~86%, ±6". |
| **Premium / dollar level** | **TRUST AS A CONSERVATIVE FLOOR** | No put skew on Bloomberg → the engine quotes ATM-IV premiums ~12–20% *below* what a real skewed chain pays. EV is understated, not inflated — safe direction. |
| **Concentration / sizing** | **TRUST ONLY IF YOU FEED IT YOUR BOOK** | The primary ranker is blind to your portfolio; R9/R10 caps fire correctly but are **opt-in** via `build_candidate_dossiers(portfolio_context=…)`. |
| **Rolling a challenged put** | **DISTRUST (silent by default)** | `suggest_rolls` returns **0** for a 20%-ITM *and* a 9%-ITM put because the default `min_net_credit=0` suppresses defensive (debit) rolls. The analysis underneath is sound — but you only see it if you know to disable the filter. |
| **Covered call on an assigned, underwater name** | **DISTRUST (empty + basis-unaware)** | Returns **0** candidates; the strike grid only enumerates strikes *below* your cost basis, then refuses them all on EV. Right action, wrong reason, no constructive alternative. |

Net: this is an **entry-selection engine you can trade**, bolted to a
**position-management surface that abdicates exactly when a wheel trader needs
it** (challenged puts, underwater assignments). Everything it *does* say is
honest; the danger is the silent zeros, which a careless trader reads as "nothing
to do" when the truth is "the default filter hid the answer."

---

## Setup (point-in-time, reproducible)

- **As-of:** 2026-03-20 (the last date the Bloomberg data supports — OHLCV / IV /
  VIX / macro all end exactly here, so look-ahead is structurally impossible).
- **Book:** $250,000 cash-secured-put wheel account.
- **Engine:** `origin/main @ 26fda24`; connector `MarketDataConnector` (bloomberg);
  `sp500_ohlcv.csv` sha256 `c3d5443158b12ec5…` (60,828,230 bytes).
- **Tape:** `get_vix_regime("2026-03-20")` → **VIX 28.97, 91st percentile,
  backwardation** (vix_3m 28.47 < spot); risk-free 3.62%; HMM regimes across the
  ranked names span bear / normal / crisis / bull_quiet. A genuinely stressed —
  not crisis — environment, ideal for stress-testing the gates.
- **Driver:** `scripts/s47_trader_session_2026_03_20.py` (observe-only; imports
  the engine, never modifies it; the WheelTracker book is in-memory).

Run: `set SWE_DATA_PROVIDER=bloomberg && python scripts/s47_trader_session_2026_03_20.py`

Every number below is from that driver's transcript. Nothing is hand-computed
except the independent BSM re-pricing (using the engine's own `option_pricer`)
and the skew estimate, both clearly labelled.

---

## Step 1 — "What should I sell puts on today?"

**Asked:** `rank_candidates_by_ev(WHEEL_UNIVERSE=46 names, dte=35, delta=0.25,
top_n=25, min_ev_dollars=0, as_of=2026-03-20)`.

**Returned:** 23 positive-EV candidates. Top of the book:

| ticker | spot | strike | %OTM | IV | premium | prob_profit | ev_$ | ev/day | roc | cvar_5 | hmm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLY | 902.21 | 832.0 | 7.8 | 0.447 | 19.997 | 0.8857 | 713.73 | 37.90 | 0.86% | −4012 | bear |
| CAT | 672.62 | 618.5 | 8.0 | 0.464 | 15.492 | 0.8857 | 444.99 | 23.63 | 0.72% | −3990 | bear |
| AVGO | 312.50 | 287.5 | 8.0 | 0.465 | 7.278 | 0.9429 | 162.20 | 9.45 | 0.56% | −5566 | bear |
| MCD | 310.26 | 296.0 | 4.6 | 0.249 | 3.760 | 0.8857 | 155.30 | 8.87 | 0.52% | −1829 | bull_quiet |
| ABBV | 204.47 | 191.0 | 6.6 | 0.352 | 3.448 | 0.9143 | 147.21 | 8.33 | 0.77% | −1291 | normal |

…tailing down to AMD (ev_$ 4.54). **23 silently dropped** — but with a full audit
trail in `df.attrs["drops"]`:

- **13 event-gated** (earnings inside the 35-DTE window ±5d): AXP, BAC, GE, GS,
  JNJ, JPM, MS, NFLX, **NKE**, T, UNH, WFC, XOM — i.e. the whole bank-earnings
  cluster. **Correct discipline.**
- **9 negative-EV**: META **−481**, ADBE **−281**, COST −102, CRM −87, NVDA −71,
  DIS −38 (+BA, PFE, QCOM). The engine refuses to sell puts on the high-IV
  momentum names even though their premium is fat — its tail model says the left
  tail eats the premium.
- **1 history-gated**: WMT (`history 70d < required 504d`) — a data artifact (WMT's
  series is short here); the 504-day survivorship gate correctly refuses it.

**Verdict: TRUST.** This is what a ranked screen *should* look like — sorted by
EV-per-day, earnings-aware, tail-aware, with a recorded reason for every
exclusion. One usability caveat: the drops live in `.attrs`, not the visible
frame, so a trader who doesn't inspect them won't know *why* JPM/NKE vanished.

---

## Step 2 — Interrogate the top names: are these real?

For the top 5 I independently re-priced each put with the engine's own
`black_scholes_price` (q=0) and checked the solved-strike delta:

| ticker | engine prem | BSM recompute (q=0) | Δ | solved-strike put delta | +3-vol skew prem | "real chain" uplift |
|---|---|---|---|---|---|---|
| LLY | 19.997 | 19.847 | −0.15 | **−0.2485** | 22.533 | **+12.7%** |
| CAT | 15.492 | 15.349 | −0.14 | **−0.2481** | 17.349 | +12.0% |
| AVGO | 7.278 | 7.219 | −0.06 | **−0.2498** | 8.151 | +12.0% |
| MCD | 3.760 | 3.585 | −0.18 | **−0.2438** | 4.510 | +20.0% |
| ABBV | 3.448 | 3.294 | −0.15 | **−0.2382** | 3.892 | +12.9% |

**Realism Check.**

| Aspect | Engine | Reality / reference | Verdict |
|---|---|---|---|
| Premium internal consistency | e.g. LLY 19.997 | my BSM q=0 = 19.847 (engine slightly higher; explained by its dividend yield, q>0 ⇒ pricier put) | **Consistent** — premiums are genuine BSM, not fabricated |
| Strike ↔ delta solve | strikes above | BSM put delta = −0.238…−0.250, target −0.25 | **Accurate** — the 25-delta strike solver is correct |
| OTM-put skew | none (ATM IV; put_iv≡call_iv on Bloomberg) | real large-cap 25Δ puts carry +3…+5 vol pts → +12–20% premium | **Mismatch (conservative)** — engine *under*-quotes premium/EV; a real fill is better, not worse |
| `prob_profit` definition | 0.8857, etc. | sum(prob_profit + prob_assignment) = 1.000 / 1.057 / 1.086 ≥ 1 | **Correct** — prob_profit = P(S_T > breakeven) properly *exceeds* 1−P(assign); it counts the premium cushion |
| `prob_profit` precision | "0.8857" | = **31/35**; 35-DTE forward dist has exactly **35** non-overlapping windows (verified) → 2.9pp grid, ±~6pp binomial SE | **Overstated precision** — honest estimate, false-precision display |

**Verdict: TRUST the strikes/premiums; DISCOUNT the premium *level* upward (skew)
and the probability *digits* (small-n).** The probabilities are honest empirical
frequencies — they capture real fat tails — but at n=35 they are coarse. A desk
would read AVGO's "0.9429" as "≈94%, give or take a few points," and would expect
to *collect more* premium than quoted once put skew is in the chain.

---

## Step 3 — Earnings edge

**Asked:** rank `[NKE, KO]` with the event gate ON and OFF. NKE reports 2026-03-31
(11 days out, inside a 35-DTE put's life); KO is a clean control.

**Returned:**
- Gate **ON** → `['KO']`; NKE dropped: `event_lockout:earnings@2026-03-31 (±5d buffer)`.
- Gate **OFF** → `['KO', 'NKE']`.

**Verdict: TRUST.** Hard, correct, exactly what a disciplined wheeler does — no
35-DTE puts written over an earnings print. The gate is a true hard stop in
`EVEngine.evaluate`, not a soft re-rank.

---

## Step 4 — Elevated-vol edge (R11)

VIX 28.97 > 25, so R11 (elevated-vol top-bin size-down) is armed. I drove
`build_candidate_dossiers` with a clean chart stub (so the chart-missing /
spot-mismatch reviewers stay quiet and R11 is observable) at two delta targets:

| delta | candidates | prob_profit | vix_level threaded | verdict |
|---|---|---|---|---|
| 25Δ (normal) | V, MA, MSFT, PG, KO, AAPL | 0.83–0.86 (< 0.90) | 28.97 | **all proceed** |
| 12Δ (deep OTM, top-bin) | V, MSFT, PG, KO | 0.91–0.97 (> 0.90) | 28.97 | **review (elevated_vol_top_bin)** |
| 12Δ | MA | **0.8857** (just under 0.90) | 28.97 | **proceed** — gate is strict-`>` |

**Verdict: TRUST.** R11 fires exactly to contract: in a VIX-29 tape it downgrades
the "can't-lose" deep-OTM premium sells (prob_profit > 0.90) to *review*, while
leaving normal 25-delta sizing alone — and MA at 0.8857 correctly slips under the
0.90 gate. This is sound: high-probability premium selling is precisely the trade
that blows up in an elevated-vol regime, and the engine knows to make you look
twice. The threading of the live `vix_level=28.97` through the fail-safe (the path
pinned in PR #309) works end-to-end.

---

## Step 5 — Concentrated book

**Asked three ways** whether the engine warns about clustering:

1. **`rank_candidates_by_ev`** — concentration columns: **NONE**; params accepting a
   book: **NONE**. Top-15 picks cluster (IT 4, Healthcare 3, ConsDisc 2…) and the
   engine says nothing.
2. **`build_candidate_dossiers` *without* `portfolio_context`** — LLY/CAT/MA/AAPL all
   **proceed**.
3. **`build_candidate_dossiers` *with* a $250k `portfolio_context`** (empty book):

| candidate | own notional | verdict | reviewer note |
|---|---|---|---|
| LLY | $83k (33% NAV) | **review** | `R9: Healthcare sector exposure would be 33.3% (limit 25.0% NAV)` |
| CAT | $62k (25%) | **review** | `R10: CAT single-name exposure would be 24.7% (limit 10.0% NAV)` |
| MA | $46k (18%) | **review** | `R10: MA single-name exposure would be 18.8% (limit 10.0% NAV)` |
| AAPL | $23k (9%) | **proceed** | (under both caps) |

**Verdict: TRUST — but it's OPT-IN and OFF by default.** The R9/R10 caps are real,
correctly computed, and conservative (single-name limit **10%**, sector **25%** —
*tighter* than `RiskManager`'s documented 20%/40% defaults). But the **primary**
"what should I sell" call can't see your book at all, and the chart-aware path
only engages the caps if you build and pass a `portfolio_context_snapshot`. A
trader who just calls `rank_candidates_by_ev` and fills the top names will load a
$83k LLY put (33% of a $250k account) with **zero** warning. Selling concentration
discipline is delegated entirely to the trader unless they opt in.

---

## Step 6 — Challenged short put → roll

A put sold ATM on 2026-02-13 is now ITM on **39 of 46** watchlist names — the
whole tape sold off into the VIX-29 spike (realistic breadth). Two cases:

**DEEP — BA**: sold 243 put (spot then 242.96), spot now **193.94 → 20.2% ITM**.

| `suggest_rolls` variant | candidates |
|---|---|
| `min_net_credit=0` (**default**) | **0** |
| `min_net_credit=−∞` (allow debit) | **16** — e.g. roll to 181/176.5/171.5 (May, 63 DTE), net **debit ~$4,200–4,500**, `roll_ev −4871 > hold_ev −4896` ⇒ `recommend=True` |

**MODERATE — HON**: sold 241.5 put, spot now **220.10 → 8.9% ITM**.

| variant | candidates |
|---|---|
| default (credit-only) | **0** |
| allow debit | **16** — roll to 209.5/205.5/201.5/196.5 (May), net debit ~$1,600–1,950, `roll_ev −1913` vs `hold_ev −2196` (**+$280 better**), new-put EV +283/+255/+221/+175, prob_otm 0.82–0.94, `recommend=True` |

**Realism Check.**

| Aspect | Engine | A disciplined desk | Verdict |
|---|---|---|---|
| Default answer on a challenged put | **0 candidates** (both 20%-ITM and 9%-ITM) | "Here are your defensive rolls and whether they beat holding" | **Wrong default** — `min_net_credit=0` hides every roll, since defensive rolls are debits |
| Roll analysis when filter relaxed | EV-grounded: net debit, roll_ev vs hold_ev, new-put EV + prob_otm | exactly the comparison a trader makes | **Sound** — the underlying math is genuinely useful |
| Strike grid considered | only OTM (lower) deltas 0.15–0.30 | also "roll out, same/near strike, for a credit" | **Structural gap** — the most common defensive roll (same-strike, longer-dated, for credit) is never enumerated |
| Deep-ITM honesty | hold_ev ≈ −$4,896; best roll ≈ −$4,871 | "this position is a near-total loss; rolling buys you ~$25" | **Honest** — it doesn't pretend a 20%-ITM put can be rescued |

**Verdict: DISTRUST the default; the engine is silent exactly when you need it.**
`suggest_rolls` is the management tool, and the only time you call it is when a
position is challenged — yet its default credit filter guarantees an empty answer
in that case. The analysis underneath (once you pass `min_net_credit=−∞`) is sound
and even appropriately bleak on the deep-ITM name. But a trader running defaults
sees "0" and concludes "no action," which is wrong.

---

## Step 7 — Assignment → covered call

Take assignment on the BA put: state → `STOCK_OWNED`, **basis 243.0** (= strike,
correct), spot **193.94** → **underwater $4,906/100sh** (before the $11.30/sh put
premium already booked to realized P&L — accounting is clean).

**Asked:** `rank_covered_calls_by_ev(BA, shares=100, dtes=(21,35,49), deltas=(.30–.15))`.

| variant | candidates |
|---|---|
| `min_ev_dollars=0` (**default**) | **0** |
| floor OFF (show refused) | **12** — strikes **205.5–230.5**, premiums $1.4–5.1, EV **−$76 to −$445** (all negative); **every strike is below the 243 basis** |

**Realism Check.**

| Aspect | Engine | A disciplined wheeler | Verdict |
|---|---|---|---|
| Sell calls here? | refuses all (all −EV in BA's 41% IV bear regime) | "don't sell cheap calls on a name that can rip" | **Right call, EV-grounded** |
| Basis awareness | every enumerated strike (205–230) is **below** the 243 basis | "never sell a call below your cost basis — it locks the loss" | **Structurally blind** — the grid is delta-off-spot, never offers a strike ≥ basis |
| Constructive alternative | none (empty) | "sell a 245–250 call for a small credit, or hold for recovery" | **Abdicates** — gives the assigned trader nothing to act on |

**Verdict: DISTRUST for the assigned-and-underwater case.** The refusal is
*defensible* (the call leg really is −EV at BA's vol), but the engine arrives there
for the wrong reason (EV of strikes that were never appropriate) and never surfaces
the one trade a disciplined wheeler would actually consider — a call at or above
the 243 basis. As a management copilot for an underwater assignment, it's empty.

---

## Where this leaves the trust question

**Trade the entry. Don't trust the management surface's silence.**

- The entry engine (rank → gate → size-down → EV-refuse) is **sound and realistic**
  on this elevated-vol tape. I would act on its ranked +EV candidates, knowing the
  premiums are a conservative floor (skew adds ~12–20%) and the probabilities are
  bands not points.
- The management tools (`suggest_rolls`, `rank_covered_calls_by_ev`) **return zero by
  default exactly in the challenged/assigned situations they exist for**, because of
  (a) a credit-only roll filter and (b) a delta-off-spot strike grid that is
  basis-unaware. The analysis underneath the roll filter is good; the covered-call
  refusal is defensible but unhelpful. A trader must know to relax the filters, and
  even then won't get the basis-aware call a desk would write.
- Concentration discipline is real but **opt-in** — fed your book, R9/R10 fire
  correctly and conservatively; left to the primary ranker, you get no warning.

This **confirms the prior** ("honest on mid-range probabilities, weak on dollar-EV
as a forecast, shaky at crisis onset") and **sharpens it**: the dollar-EV weakness is
specifically a *premium-level* understatement from missing skew (a *conservative*
error), the probability honesty comes with *small-sample coarseness*, and the
sharpest gap is not crisis pricing but **position management ergonomics** — silent
defaults at the moment of maximum need.

---

## Findings (observe-only; no engine change in this PR)

- **F-S47-1 (management silence — HIGH):** `suggest_rolls` default `min_net_credit=0`
  returns 0 for challenged puts (the only puts you'd roll). Consider defaulting to
  show debit rolls flagged, or surfacing "N credit-positive rolls; M debit rolls
  available."
- **F-S47-2 (basis-unaware CC grid — HIGH):** `rank_covered_calls_by_ev` enumerates
  strikes by delta off current spot, so on an underwater assignment it only ever
  offers (and then refuses) strikes below basis. It never proposes a call ≥ basis.
- **F-S47-3 (roll strike grid — MED):** neither roller enumerates the same/near-strike
  roll-out-in-time-for-credit — the canonical defensive roll.
- **F-S47-4 (concentration opt-in — MED):** the primary `rank_candidates_by_ev` is
  portfolio-blind; R9/R10 only engage via `build_candidate_dossiers(portfolio_context=…)`.
  No warning that you're about to load 33% of NAV into one name.
- **F-S47-5 (false-precision probabilities — LOW):** `prob_profit`/`prob_assignment`
  are k/N empirical counts (N=35 at 35-DTE, 59 at 21-DTE, 25 at 49-DTE), displayed to
  4 decimals. Worth surfacing N or a CI so the coarseness is visible.
- **F-S47-6 (skew-blind premium — LOW/known):** no put skew on Bloomberg → premiums/EV
  ~12–20% below a real chain. A *conservative* bias, but EV magnitudes shouldn't be
  read as fill-accurate. (Consistent with S29.)
- **F-S47-7 (drops not surfaced — LOW):** the 23 exclusion reasons live in
  `df.attrs["drops"]`, invisible in the returned frame.

None of these is a §2 breach — every gate is downgrade-only and no negative-EV
candidate was rescued. They are *answer-quality / ergonomics* findings about the
management phase.

---

## Reproduce

```
cd <repo>            # origin/main @ 26fda24 or later
set SWE_DATA_PROVIDER=bloomberg
python scripts/s47_trader_session_2026_03_20.py
```

Deterministic (forward-dist `seed=42`, fixed Bloomberg CSVs). Prints the full
transcript and writes a JSON artifact to `%TEMP%/s47_session_2026-03-20.json`.
The driver is observe-only; it never touches `engine/`.

## Unresolved / handoff

- The seven findings are management-phase ergonomics, candidates for a future
  *behaviour* PR (e.g. a basis-aware CC strike grid, a defensive-roll default). Each
  must stay downgrade-only and route through `EVEngine.evaluate` (§2).
- A natural follow-up usage scenario: the same session on a **low-VIX** date to
  confirm R11 goes dormant (< 25) and the management tools behave identically — i.e.
  the silence is structural, not regime-specific.
- This session used the bloomberg connector (no real chains). Re-running the
  management steps on the **theta** provider (real chains, real skew) would test
  whether F-S47-2/3/6 change when actual OTM-put quotes are available.
