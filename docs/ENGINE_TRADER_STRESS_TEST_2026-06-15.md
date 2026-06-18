# Engine Trader Stress Test — 2026-06-15

**Method:** role-played a professional cash-secured-put wheel trader using the
engine for real decisions, then checked every output against an independent
calculation or against *what actually happened*. Bloomberg provider (committed
data, ends 2026-03-20). All scans pinned to historical `as_of` so predictions
could be replayed against realized OHLCV. Out-of-sample replay = held-to-expiry
P&L on the engine's own strike/premium, 35-DTE 25-delta short puts.

> Bottom line: the **machinery is trustworthy** (pricing exact, fully
> deterministic) and the engine is **honestly calibrated in calm tape** — but it
> is **dangerously over-confident in crises**, its **tail model understates real
> single-name blowups by 3–4.5×**, its reported edge is **mid-priced (optimistic
> on execution)**, and its book-builder **concentrates 35–45% of NAV on one
> name** with the protective caps off by default. Used naively with real money it
> would lead a trader into exactly the trades that blow up.

---

## What is reliable (verified, trust it)

1. **Core pricing math — exact.** Independently re-solved the 25-delta strike and
   BSM put premium from scratch: engine `K=144.50 / prem=1.922 / Δ=−0.255` vs my
   `K*=144.31 / 1.921` for XOM (JPM, MSFT match to the mil). Strikes round to real
   listed strikes. The BSM layer is correct and realistic.
2. **Determinism — perfect.** Same scan repeated, and with **reversed ticker
   order**, produced byte-identical `ev_dollars` for every name. Reproducible and
   order-invariant despite the HMM issue below. A trader gets the same answer
   every time.
3. **Calm-regime calibration — good, even conservative.** OOS replay, 2023–2026,
   40 large-caps, n=747: the 0.80–0.90 `prob_profit` bin realized 0.806 vs 0.828
   predicted (−2.2pp); lower bins were *under*-confident (+4 to +12pp). Traded
   region (ev>0, n=149): predicted 0.839 vs realized **0.852** win-rate, mean
   **+$0.93/share**. In normal markets the engine's probabilities are honest.
4. **Graceful failure on bad inputs.** Invalid deltas (0.49, 0.99, −0.1), DTE 0/1,
   unknown/empty tickers all → clean drops, no crashes, no garbage. Size scales
   linearly without overflow (1000 contracts → ev ×1000 exactly).
5. **The engine's own docs are honest** about the defensive-sleeve framing and the
   prob_profit over-confidence — this test confirms those caveats rather than
   contradicting them.

---

## What is NOT reliable / not realistic (the real-money risks)

### 1. prob_profit is severely over-confident in a crisis (regime-dependent)
OOS replay on 2020 crash + 2022 bear, n=399 — **every** bin over-confident:

| prob_profit bin | n | predicted | realized | miscal |
|---|---|---|---|---|
| <0.70 | 20 | 0.681 | 0.300 | **−38.1pp** |
| 0.70–0.80 | 124 | 0.758 | 0.653 | −10.5pp |
| 0.80–0.90 | 190 | 0.841 | 0.700 | **−14.1pp** |
| 0.90–0.95 | 41 | 0.922 | 0.854 | −6.9pp |
| 0.95–1.0 | 24 | 0.974 | 0.917 | −5.8pp |

When it says "84% safe" in a crisis, reality is 70%. The calm sample masks this
entirely — **the calibration is a regime artifact, not a stable property.** R11
(VIX>25 + prob_profit>0.90 → review) only guards the top bin; the −14pp damage in
the **0.80–0.90 band (n=190) is below R11's 0.90 threshold and unprotected.**

### 2. The crisis "+EV recommended" trades LOSE money on average
Crisis traded region (ev_dollars>0, n=180): **win-rate 0.811 but mean P&L
−$81/contract.** An 81% win rate with a negative mean — the losers are huge. A
trader following the engine's own buy signal through 2020/2022 loses money.

### 3. The tail model (cvar_5) understates real single-name blowups by 3–4.5×
- **Realized loss breached the modeled `cvar_5` in 10% of crisis candidates** —
  `cvar_5` is the *expected shortfall beyond the 5% VaR*, so breaches should be
  rare, not 1-in-10.
- **Worst calm-window trade — UNH @ 2024-11-15:** engine *recommended* it
  (`ev_dollars=+199`, `prob_profit=0.829`, `cvar_5=−$1,989`). UNH fell −15.6% in
  35 days → **−$6,035/contract — 3× worse than the modeled 5% tail.**
- **Worst crisis trade — BA @ 2020-02-20:** realized **−$13,571 vs modeled
  `cvar_5` −$3,016 (4.5× too thin).**

Root cause: the empirical forward distribution can't contain a move it hasn't
recently seen, so the tail it prices is structurally too thin exactly when
idiosyncratic risk hits. A trader sizing by `cvar_5` is systematically
under-hedged on the names that matter.

### 4. Reported edge is mid-priced — optimistic on execution
The engine reports the **mid** premium and computes EV from it. A trader sells at
the **bid**: an 8% haircut (~$22/contract on a 10-name scan). Against the
traded-region edge of +$0.93/share, the bid haircut (~$0.22) alone eats **~24% of
the edge** — before the omitted exit-leg cost (D19, ~$1–4/contract) and before
real bid-ask widening on the illiquid high-strike names that dominate the book.
Bloomberg's zero-skew IV compounds this: the OTM put is priced at ATM IV, so the
engine both under-collects the real (skewed) premium and under-prices the real
downside the market is charging for.

### 5. select_book builds a dangerously concentrated book
At $200k it deployed **97% of NAV into 4 names: GS 44.8%, CAT 34.8%, ABBV 10.6%,
XOM 7.2%.** Three of four breach the R10 10% single-name cap — but **R9/R10 are
dormant by default** (`require_ev_authority=False`), so the builder happily ships
it. It front-loads the highest-collateral names with no concentration control —
maximum exposure on exactly the single names where finding #3 shows the tail model
fails worst.

### 6. Fabricated certainty at long horizons + the prob_profit trap
- `dte_target=400` → **`prob_profit=1.000` with a *positive* `cvar_5`** (the 5%
  worst case shown as a *profit*). The engine claims certainty and zero tail risk
  at long horizons, and does not refuse out-of-wheel-range DTEs.
- At baseline, AAPL showed **`prob_profit=0.857` with *negative* EV** — a
  high-confidence-but-fat-left-tail trade. `prob_profit` alone is a trap; only the
  EV (which a naive trader ignores in favor of the friendlier "85% win" number)
  catches it.

### 7. The regime signal driving position size comes from non-converged fits
Every HMM fit in every scan returned **`hmm_converged=False`** (n_iter=20 too low
on 504-day windows). The "bull_quiet / bear / normal" labels that scale the
position multiplier are reads off non-converged posteriors — and the label is
assigned by sorted state position, not the fitted regime (see the adversarial
review). Deterministic, but not trustworthy as a regime read.

---

## The composite failure mode (how this hurts a real account)

Naive use — raw ranker + `select_book`, default settings — chains the failures:
**over-confident `prob_profit` → +EV recommendation → too-thin `cvar_5` →
`select_book` concentrates 35–45% NAV on the name → no R9/R10 cap → a single
−$6k/−$13k idiosyncratic blowup.** Every protection that would break the chain
(R10 single-name cap, R11 top-bin size-down) is off or partial by default. The
engine is a **good ranker wrapped in an over-confident risk model with its safety
rails unarmed.**

## What a trader should actually do with it
- Use it for **selection (top-K beats random, tail-avoidance), not for sizing or
  probability bets.** Trust the ranking, distrust the `prob_profit` headline.
- **Arm R10 (`require_ev_authority=True`) and R11**, and cap single-name NAV
  manually — `select_book` will not.
- **Haircut every premium to the bid** and subtract exit costs before believing
  the edge.
- **Treat `cvar_5` as a floor on a floor** — multiply by ~3–4× for single-name
  idiosyncratic risk, especially in elevated-VIX tape.
- **Distrust all probabilities when VIX is elevated** — the calibration inverts.

*Evidence: independent BSM re-derivation; OOS forward-replay calibration (n=747
calm, n=399 crisis); per-trade tail-breach analysis; determinism + edge-input
probes. Methods reproducible against committed Bloomberg data at the pinned
`as_of` dates above. Survivorship caveat: the universe is current S&P members, so
realized win-rates here are if anything optimistic.*
