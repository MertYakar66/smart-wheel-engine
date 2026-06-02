# Premium-correction pilot (observe-only)

**Status:** harness + split layer landed and validated end-to-end; full
3-name run pending larder band coverage. **Scope:** read-only. Touches no
decision-layer file (`ev_engine.py` / `wheel_runner.py` /
`candidate_dossier.py`); calls the authoritative EV path
(`WheelRunner.explore_ticker` → `rank_candidates_by_ev` → `EVEngine.evaluate`)
read-only and joins the real Theta larder mid afterwards.

Code: `studies/premium_correction/` (`pilot.py`, `splits.py`).
Tests: `tests/test_premium_correction_pilot.py`.
Outputs: `studies/premium_correction/output/`.

---

## 1. Why this exists

On the Bloomberg/synthetic path the engine constructs the short-put premium
as `premium = BSM(iv)` and computes `fair = BSM(iv)` from the **same** iv, so
`edge_vs_fair = premium − fair` is **structurally zero** (annotated in
`wheel_runner.py` at the `premium_source = "synthetic_bsm"` stamp). The
larder's real unlock is therefore **real premiums**, not skew signals layered
on a synthetic premium. This pilot measures what swapping the synthetic
premium for the real Theta EOD mid does to `edge_vs_fair`, holding the risk
side fixed.

## 2. What the number is — and what it is NOT (labeling discipline)

The first-cut wiring changes **only** the premium and leaves `fair = BSM(iv)`
untouched. Then, identically:

```
edge_vs_fair  ==  real_mid − BSM(iv)  ==  premium correction
```

For OTM puts this is **positive largely because real OTM implied vol exceeds
the flat ATM iv** the engine prices against — i.e. it is **skew-driven**. It
is reported as **"how much the engine under-prices the premium,"** *not* as
"the volatility risk premium." The true economic VRP requires a proper
**reference vol** for `fair` (realized-vol or a forecast), which is a
**separate, second** measurement — deliberately not done here. Reading this
headline as "free money" is the seductive half-truth the pilot exists to
guard against; §3 is the guard.

Validation anchor (post-split band, clean join): AAPL 35-DTE 0.25Δ short put,
as_of 2024-09-06 — synthetic `BSM(iv)=2.869`, real mid `3.300` →
**+20.8%** under-pricing. (The split-bug version of this same join printed a
bogus **−99%** by matching an adjusted strike to a raw strike — see §5.)

## 3. The deliverable — Refinement 2 cross-plot

Per candidate we also compute, on the **same** contract:

- `mkt_prob_itm` — risk-neutral `P(S_T < K)` under the **contract's own**
  implied vol (inverted from the real mid). The market's priced downside.
- `eng_prob_itm` — the engine's **empirical** `prob_assignment` from its
  forward distribution. The risk the engine's model sees.
- `calib_gap = mkt_prob_itm − eng_prob_itm`. **> 0 ⇒ the market prices more
  downside than the engine's empirical distribution sees** (engine
  over-confident there).

The cross-plot is **premium under-pricing (x)** vs **`calib_gap` (y)**. If the
strikes with the **largest** premium correction cluster in the **upper-right**
(big correction *and* positive calib_gap), repricing the premium alone books
`edge_vs_fair` as free money on exactly the strikes where the engine's risk
model is most over-confident → the honest wiring is **reprice-and-reshape**,
not reprice-premium. If big-correction strikes sit at `calib_gap ≈ 0` (benign
empirical tails that agree with the market), reprice-premium is closer to
honest. **The cross-plot is the result that decides the §2 wiring arc.**

## 4. What this pilot can and cannot settle

**Can (3 post-split names, 2024-09 → 2026-03):**
- Refinement 1 — the **magnitude** and sign of the premium correction, by
  name / delta / DTE.
- The **cross-sectional** form of Refinement 2 — at a point in time, do
  fat-premium strikes sit in benign empirical tails?

**Cannot:**
- **Verdict-flip rate.** That is cross-sectional and needs ~15 names; **do
  not quote a flip %** off three. Three names answer magnitude +
  risk-alignment only.
- **Crisis-onset tail amplification** (the I1/I3-E worst case, where the
  empirical forward distribution **lags the regime** at a vol spike and is
  most over-confident exactly where put skew is fattest). That only surfaces
  across a **stress window**. If 2024-09 → 2026-03 is benign, a "risk looks
  aligned" reading here is **NOT** the final word on Refinement 2 — that needs
  the fuller study spanning **2020 / 2022**. See `[[theta-option-history-all-strikes]]`.
- **Universe-representative magnitude.** TSLA/NVDA are **high-skew outliers**;
  their correction is nearer an upper bound than the universe median (a
  JPM/XOM will be milder). Treat the pilot magnitude as **suggestive, not
  representative**, until the broader set lands.

## 5. Two correctness traps caught before any number (the discipline)

1. **Data overlap, not expiration count.** The engine needs **≥504 trading
   days of OHLCV before `as_of`** to build the forward distribution. With
   Bloomberg starting **2018-01-02**, the engine-usable window is
   **2020-02 → 2026-03**, *not* 2018 → 2026. The larder's 2018–19 expirations
   — however many — **cannot feed wheel-entry evaluation at all**. The
   "N expirations done" coverage count was real but counted unusable cells.

2. **Split-space join.** Engine spot/strike are split-**adjusted** (Bloomberg);
   larder strikes are **raw** (as-listed). Joining the two directly mis-matches
   across any split between `as_of` and today (the −99% above). The pilot band
   was chosen **post-all-splits** (AAPL 4:1 2020-08, TSLA 5:1 2020-08 + 3:1
   2022-08, NVDA 4:1 2021-07 + 10:1 2024-06) so the join is split-free; the
   split layer (`splits.py`) is unit-tested against the AAPL 4:1 split
   independently, since the band itself only exercises the identity path.

## 6. Structural notes for the full study (not just the pilot)

- **Do not lower the 504d history gate to force more data.** The forward
  distribution is the exact risk axis Refinement 2 measures; thinning it
  corrupts the thing under test.
- **The deferred 2016–17 *option* extension is moot for engine evaluation
  unless the *OHLCV* is also extended back** to 2014 (504d before a 2016
  `as_of`). Worth knowing before anyone spends days pulling 2016–17 options.
- The full study should span a **stress window (2020/2022)** to test the
  crisis-onset amplification §4 flags, and broaden to **~15 names** for
  verdict-flip and representativeness. That run needs the larder to reach
  those expirations and a **split layer** (already built + tested) because the
  pre-2024 band is not split-free.

## 7. Reproduce

```
python -m studies.premium_correction.pilot
```

Writes `pilot_records.csv` (per-candidate), `pilot_summary.md` (per-name
correction + calib_gap distributions, clean joins only), and
`crossplot_correction_vs_calibgap.png`.
