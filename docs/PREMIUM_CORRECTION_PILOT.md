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
as_of 2024-09-06 — engine solves strike 210.5, snaps to the **listed** 210
strike (exp 2024-10-11); `BSM(iv)` recomputed **at that listed contract** is
`2.733`, real mid `3.300` → correction `+0.567/sh` = **+20.8%** under-pricing.
(The `fair` must be recomputed at the listed strike — the engine's own fair at
its un-snapped 210.5 strike is `2.869`, a different contract; conflating the two
gives an inconsistent figure.) The split-bug version of this same join printed a
bogus **−99%** by matching an adjusted strike to a raw strike — see §5.

## 3. The deliverable — Refinement 2, physical-vs-physical

The decision (reprice-premium vs reprice-and-reshape) hinges on **engine
prediction vs reality**, so the deliverable axis must be **physical-vs-physical**:

- `eng_prob_itm` — the engine's **physical** `prob_assignment` from its
  empirical forward distribution. The risk the engine's model *predicts*.
- `realized_itm` — did the contract **actually** finish ITM? `1` if the
  split-adjusted Bloomberg close at the real expiry is below the (adjusted)
  strike, else `0`. The risk that *actually happened*. Resolved only when
  Bloomberg has the terminal price (`real_exp ≤ 2026-03-20`); the last ~3
  months of the band are unresolved and excluded.

Two views:

1. **Reliability curve** — bin candidates by `eng_prob_itm`, plot predicted vs
   realized assignment frequency against the 45° line. Points **above** the
   line ⇒ the engine **under-sees** realized risk.
2. **The deliverable cross-plot** — bin by **premium under-pricing (x)**, plot
   **`realized_itm − eng_prob_itm` (y)**. If the **high-correction** (fat-skew)
   bins show a **rising, positive** gap (realized assignment exceeds what the
   engine predicted), then repricing the premium alone books `edge_vs_fair` as
   free money on exactly the strikes where the engine **under-models the
   realized tail** → the honest wiring is **reprice-and-reshape**. If the gap is
   ~0 across correction (engine calibrated even where premium is fat),
   reprice-premium is closer to honest. **This is the result that decides the
   §2 wiring arc.**

### Independence: cluster, don't count contracts (the second trap)

Each (ticker, as_of) yields **many** (delta, dte, strike) contracts that all
resolve against the **same** terminal price path — they are **pseudo-replicates,
not independent trials**. Counting 155 contracts as 155 binomial draws makes a
naive Wilson interval **far too narrow**. So the deliverable uses a **cluster
(block) bootstrap** over `(ticker, as_of)` events: CIs are computed by resampling
whole clusters, `n_clusters` is reported alongside `n`, and a bin is
`trustworthy` only when it clears **both** `n ≥ MIN_BIN_N` (30) **and**
`n_clusters ≥ MIN_CLUSTERS` (8). The summary also prints the **per-name** gap,
because a single name's directional run (one stock that fell over the window)
pools into a fake cross-sectional signal otherwise. The preliminary 3-name run
demonstrated exactly this — see §4.1.

### Why NOT a risk-neutral comparison (the first trap)

An earlier cut used `calib_gap = mkt_prob_itm − eng_prob_itm`, where
`mkt_prob_itm = N(−d2)` at the contract's implied vol. That is **wrong as a
calibration measure**: `mkt_prob_itm` is **risk-neutral (Q)** while
`eng_prob_itm` is **physical (P)**, and for OTM equity puts `Q > P` **by
construction** — that wedge **is the risk premium**, positive even when the
engine's physical model is perfectly calibrated. Worse, both `Q`-implied
`P(ITM)` and the premium correction grow with skew, so plotting one against the
other is **x against a transform of x** — a tidy upper-right cluster that
"confirms" reprice-and-reshape regardless of the truth. `Q − P` is retained in
the records as **`risk_premium_wedge`** (the *apparent* free edge the
premium-only wiring would perceive) for context, but it is **not** a deliverable
axis. Only realized-vs-predicted distinguishes genuine under-modeled risk from a
normal risk premium.

## 4. What this pilot can and cannot settle

**Can (3 post-split names, 2024-09 → 2026-03):**
- Refinement 1 — the **magnitude** and sign of the premium correction, by
  name / delta / DTE.
- The **cross-sectional** form of Refinement 2 — across the band's resolved
  contracts, is the engine's predicted assignment **mis-calibrated against
  realized outcomes more in the high-premium-correction strikes** than the
  low-correction ones? (Caveat: predicted `prob_assignment` is at the engine's
  solved strike/DTE; realized is at the snapped listed contract — held within
  4% strike / 5 days by the join-quality gate.)

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

**Known confounds in the realized-vs-predicted *level* (audit-surfaced):**
- **Horizon-units bug (DECISIONS D21).** The engine's `prob_assignment` forward
  distribution indexes **trading-day** bars while callers pass the option's
  **calendar** DTE, so the engine's predicted horizon is **~46% too long** →
  `prob_assignment` is **inflated**. The pilot consumes `prob_assignment` as-is,
  so the *level* of the gap (and any "engine conservative / over-predicts"
  reading) is **partly a D21 artifact, not risk-model shape**. D21 is a deferred
  coordinated re-baseline, not a point fix — the pilot must **not** patch it; the
  cross-sectional *shape* (gap vs correction) is more robust to a roughly uniform
  horizon bias than the level is.
- **Deep-OTM selection bias.** Far-OTM puts often have `bid=0`, so
  `spread_pct=(ask−0)/(ask/2)=200% > MAX_SPREAD_PCT` and are dropped by
  `join_ok`. That trims exactly the **fattest-skew tail** the study targets —
  biasing the measured signal **toward the null** (against reprice-and-reshape).
- **Residual x–y coupling via strike depth.** `correction_pct` grows with OTM
  depth; `eng_prob_itm` falls with OTM depth. The coupling is far weaker than the
  repudiated Q−P tautology, but the high-correction bin is also the
  deep-OTM/low-predicted bin, so read the *gap* (not realized alone) and lean on
  the cluster-robust CI.

### 4.1 Preliminary 3-name result — a mirage, dissolved by clustering

The first preliminary run (≈200 resolved contracts, AAPL/NVDA/TSLA, calm
2024–25) produced a **striking but false** reprice-and-reshape signal, and
dissolving it is itself the finding:

- **Naive read (independence-assuming Wilson):** the high-correction bin showed
  realized assignment **0.42 vs predicted 0.20, gap +0.22, CI [+0.06, +0.39]
  clearing 0**, rising monotonically — textbook "engine under-sees risk where
  premium is fat."
- **Why it's false:** (1) **pseudo-replication** — those ≈200 contracts are
  only **~14 independent (ticker, as_of) events** (the high-correction bin: 5–6
  clusters); the naive CI counted correlated contracts as independent draws.
  (2) **single-name direction** — the entire signal is **AAPL** (which drifted
  down in the resolved window); **NVDA −0.16, TSLA −0.30** show the *opposite*
  (engine conservative).
- **Cluster-robust read:** resampling whole `(ticker, as_of)` clusters, the
  high-correction gap collapses to **+0.05, CI [−0.25, +0.61]** — every
  high-correction bin falls below the 8-cluster floor and is flagged
  `not signal`. Verdict: **no calibration failure established; the apparent
  signal is within cluster-robust noise.**

Refinement 1 (premium correction is real and positive — medians roughly
AAPL ~+16%, NVDA ~+5%, TSLA ~+1% of BSM) holds. Refinement 2 is **inconclusive
by construction** at 3 names / ~14 clusters — it **requires the ~15-name set**
(so direction averages out and clusters multiply) **and** a 2020/2022 stress
window. The preliminary's value was negative-space: it proved the guard catches
a confident-wrong result that two earlier traps (Q-vs-P, then small-n) would
have let through.

## 5. Two data-level traps caught before any number (the discipline)

*(The two methodology/statistics traps — risk-neutral-vs-physical (§3) and
pseudo-replication (§3, §4.1) — were caught at/after first numbers; these two
are the data-level ones caught before any number ran.)*

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

Writes (to `studies/premium_correction/output/`, gitignored — regenerable and
moving with larder coverage): `pilot_records.csv` (per-candidate),
`pilot_summary.md` (per-name correction + the cluster-robust Refinement-2
read), `reliability_bins.csv` + `crossplot_bins.csv` (binned data of record
with Wilson **and** cluster-robust CIs + `n_clusters`), and
`refinement2_calibration.png` (reliability curve + the deliverable cross-plot,
low-cluster bins faded). The PNG renders only if matplotlib is installed; the
CSVs are always written.
