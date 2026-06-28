# Heavy-verify — Data-wiring accuracy + engine output-realism reliability (Mac terminal)

**Campaign:** #436 · **Branch:** `claude/mac-data-wiring-reliability` · **Started:** 2026-06-27
**Terminal:** Mac (fresh clone, no real-premium rail) · **Mode:** validation-only (measure & report; no engine-behaviour edits)
**Data path / provider:** `SWE_DATA_PROVIDER` unset → **bloomberg** (committed CSVs under `data/bloomberg/`); connector = `MarketDataConnector`.

> Every claim below carries the data path and, where it is a statistical estimate,
> a confidence interval. Per-check JSON is persisted under
> `docs/verification_artifacts/data_wiring_2026-06-27/` **before** any pretty-print,
> so a console-encoding crash never loses compute. `PYTHONIOENCODING=utf-8` set for all runs.

---

## §1 Bring-up (green baseline)

| Step | Result |
|---|---|
| Python | 3.12.2 (repo targets 3.12) in `.venv` |
| Deps | `requirements.txt` + dev extras (`hypothesis`, `requests-mock`, `pytest-asyncio`) — the test extras live in `pyproject [project.optional-dependencies].dev`, not `requirements.txt` |
| Provider | `SWE_DATA_PROVIDER` **unset → bloomberg** (logged; silent selection is a known bug source). `type(WheelRunner().connector).__name__` = `MarketDataConnector` ✓ |
| 5-ticker smoke (CLAUDE.md §4.2) | healthy — `ev_dollars` / `iv` / `premium` all non-null. **4/5 rows** returned (JPM filtered out — investigated in W2). |
| Green baseline | `pytest tests/ -m "not backtest_regression"` (the per-PR CI lane, `.github/workflows/ci.yml`) = **3275 passed, 33 skipped, 17 xfailed, 0 failed** (349 s). |

**Note on the 4 backtest-regression failures.** A bare `pytest tests/` reports 4 failures
(`test_backtest_regression.py::…[s27/s32/s34/s35]`). These are the slow (S34 ~3.5 h) snapshot
reproducers that the CI lane **deselects** via `-m "not backtest_regression"` and that fail in this
sandbox on a missing optional `typer` dependency — **not a code regression**. The authoritative
green baseline is the CI-lane count above. Every PR in this campaign keeps that lane green.

---

## §W1 — Data-wiring accuracy audit

**Driver:** `scripts/audit_data_wiring.py` → JSON: `w1_ohlcv.json`, `w1_vol_iv.json`,
`w1_treasury.json`, `w1_generic_sources.json`, `w1_summary.json`.

### Per-source pass/fail table

| Source | Connector accessor | Rows / coverage | Monotonic | NaN / sentinel | Units | Verdict |
|---|---|---|---|---|---|---|
| OHLCV | `get_ohlcv` | 2117 rows/ticker, 2018-01-02→2026-06-04 | ✅ per ticker | 0 NaN/non-positive close (sample); OHLC invariant `high≥max(o,c,l)` holds 0 violations | split-adjusted $ | ⚠️ **PASS except split-scale defect (BKNG, CVNA)** |
| vol/IV | `get_iv_history` | 1,037,278 raw rows; 2,062,702 served IV cells | n/a | raw carries 17 sub-3 + 5,910 NaN cells; **served band 100% clean** — every served IV ∈ (3.0, 10000] | PERCENT | ✅ PASS |
| Treasury / RFR | `get_risk_free_rate` | rate_3m 8,455 rows, **1994-01-03→2026-06-05** | ✅ | none | decimal (÷100, D20) | ✅ PASS |
| Dividends | `get_dividends` | AAPL 91 rows | ✅ ex_date | 1 negative `dividend_amount` (AAPL) — flagged below | $ | ✅ PASS (minor) |
| Corp actions | `get_corporate_actions` | AAPL 6 disruptive | ✅ eff_date | — | ratio/amount | ✅ PASS |
| Fundamentals | `get_fundamentals` | snapshot dict, 13 keys | n/a | — | IV PERCENT (`implied_vol_atm`=25.57), beta raw | ✅ PASS |
| Credit risk | `get_credit_risk` | dict: `sp_rating`, `altman_z_score`, `interest_coverage_ratio` | n/a | — | — | ✅ PASS |
| VIX | `get_vix` / `get_vix_regime` | 9,200 rows, 1990-01-02→2026-06-04, range [9.14, 82.69] | ✅ | — | index points | ✅ PASS |

### Defect list (copy-pasteable reproductions)

#### D-W1-1 — OHLCV split-scale discontinuity at the 2026-03-23 data splice (BKNG ÷25, CVNA ÷4.7) — **CONFIRMED**

The OHLCV monolith is a splice of an older split-adjusted pull (history) and a recent slice
(≥ 2026-03-23). Names that split **between** the two pulls carry the split adjustment **only on the
recent slice**, producing a one-day scale break at 2026-03-23 plus an early-application window where
served prices are post-split-scaled but the real market was still pre-split.

```python
from engine.data_connector import MarketDataConnector
c = MarketDataConnector()
b = c.get_ohlcv("BKNG")["close"]
print(b.loc["2026-03-20"], "->", b.loc["2026-03-23"])   # 4286.81 -> 175.87  (ratio 0.041 ≈ 1/25)
# BKNG 25:1 split EFFECTIVE 2026-04-06, but applied 2026-03-23 (~2 weeks early)
v = c.get_ohlcv("CVNA")["close"]
print(v.loc["2026-03-20"], "->", v.loc["2026-03-23"])   # 283.28 -> 59.92  (ratio 0.212 ≈ 1/4.7)
# CVNA 5:1 split EFFECTIVE 2026-05-08, applied 2026-03-23 (~6.5 weeks early)
```

- **Pre-2026-03-23** served prices are the *real* point-in-time (unadjusted) market prices.
- **From 2026-03-23** they are post-split-scaled.
- **Effect:** (a) an internal discontinuity — a fake **−96%** (BKNG) / **−79%** (CVNA) single-day
  return on 2026-03-23 that poisons any realized-vol / return / tail computation crossing that date;
  (b) a window `[2026-03-23, eff_date)` where served spot is **25× / 4.7× too low** vs the true
  PIT market, so a backtest with `as_of` in that window mis-scales spot → strike → premium → `ev_dollars`.
- **Reach:** only at historical `as_of` ∈ the affected window or backtests crossing 2026-03-23. The
  default ranker uses the latest frontier (2026-06-04), where the post-split scale is *correct*, so
  **current/live rankings are unaffected**.
- **Fix class:** **data regeneration** (re-pull OHLCV with consistent full-history adjustment, or apply
  the adjustment only from the effective date) — *not* a decision-trio fix and not a connector-behaviour
  edit this campaign is permitted to make. Filed as a GitHub issue for the rail/data owner. Pinned by a
  strict-xfail test in `tests/test_w1_data_wiring.py`.

#### D-W1-1b — NFLX ~10× scale — **REFUTED as a corruption**

NFLX 2018-01-02 served close = **$20.107** (real unadjusted ≈ $201.07). This is **not** a per-ticker
corruption: the **entire** NFLX series is uniformly back-adjusted ÷10 for the 2025-11-17 10:1 split
(no internal discontinuity — a universe sweep finds **zero** >2× daily jumps in NFLX). This is standard
Bloomberg split-adjusted convention and is internally consistent: strikes, premiums and spot all scale
together, so `prob_profit` / IV / returns are unaffected and only the *absolute* dollar magnitude of a
pre-2025-11-17 NFLX backtest is ÷10 — the same property every split in the dataset exhibits (e.g. AAPL
pre-2020 is ÷4). **Not a defect.**

#### Real-move discontinuities (NOT defects, recorded for completeness)

The universe sweep (511 tickers) found **6** names with a >2× single-day close move. Two are the split
artifacts above; the other four are genuine market events:

| Ticker | Date | Move | Cause |
|---|---|---|---|
| GL (Globe Life) | 2024-04-11 | −53% | Fuzzy Panda/Viceroy short-seller report |
| OXY | 2020-03-09 | −53% | Saudi-Russia oil price-war crash |
| TRGP | 2020-03-09 | −53% | same oil crash |
| PCG (PG&E) | 2019-01-14 | −52% | wildfire-liability bankruptcy announcement |

#### D-W1-2 — one negative AAPL `dividend_amount` row — **minor, flagged**

`get_dividends("AAPL")` has 1 row with a negative `dividend_amount`. Low-severity (dividends feed the
ex-div lockout / carry, not a hard EV gate), recorded for the data owner; not pinned.

### Verified properties (pinned green in `tests/test_w1_data_wiring.py`)

- Served IV band is always `(3.0, 10000]` (the `_clean_vol_iv_inplace` gate) on a sampled set — raw
  sub-3 / sentinel cells never leak to a consumer.
- The `134217.7` deep-IV sentinel and the 10000-floor: **0** occurrences in the committed monolith
  (the sentinel lives only in the deep/delisted gz panels, absent on this clone), and the served band
  is clean regardless.
- OHLCV post-rename invariant `high ≥ max(open, close, low)` holds with 0 violations on the sample.
- Treasury `rate_3m` coverage spans the full feasible OHLCV window (starts 1994 ≤ 2018).

---

## §W2 — Engine output realism

**Driver:** `scripts/audit_output_realism.py` → `w2_output_realism.json`.
Full-universe `rank_candidates_by_ev` at 5 historical `as_of` dates spanning regimes;
Greeks recomputed per candidate via `engine.option_pricer.black_scholes_all_greeks`
against `docs/GREEKS_UNIT_CONTRACT.md`.

### Realism table (per regime)

| Regime | as_of | VIX | RFR 3m | ranked | prem/spot med | IV decimal (min–max) | prob_profit max | non-finite | greek violations |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| calm | 2021-06-15 | 17.0 | 0.01% | 358 | 1.07% | 0.134–0.548 | 0.958 | **0** | **0** |
| calm | 2024-01-16 | 13.8 | 5.36% | 71 | 1.09% | 0.146–0.463 | 0.886 | **0** | **0** |
| crisis | 2020-03-23 | 61.6 | −0.04% | 191 | 3.70% | 0.478–2.737 | 0.998 | **0** | **0** |
| bear | 2022-06-16 | 33.0 | 1.49% | 329 | 1.94% | 0.231–1.412 | **1.000** | **0** | **0** |
| bear | 2022-10-14 | 32.0 | — | 37 | 1.79% | 0.246–0.921 | 0.941 | **0** | **0** |

**Verdict: outputs are realistic.** Across all 5 regimes and ~986 ranked candidates:
**0** non-finite values, **0** Greek-contract violations (every 25-delta short put solves to
`delta ∈ [-1,0]`, `gamma ≥ 0`, `vega ≥ 0`), **0** `prob_profit`/`prob_assignment` outside `[0,1]`,
**0** served IV outside the decimal band `(0.03, 5.0]`, **0** `premium/spot` ≥ 0.5, **0** `spot < 1`.
Magnitudes track the regime correctly: `premium/spot` rises 1.1% (calm) → 1.9% (bear) → **3.7%
(crisis)**, and IV ceilings rise 0.55 (calm) → 1.41 (bear) → **2.74 (crisis)** — exactly the
direction option-seller economics predict. RFR is the real PIT rate each day (0.01% ZIRP 2021,
5.36% in 2024, slightly negative at the COVID trough). Drops bucket cleanly into
`event` / `history` / `data` (no silent vanishing — `frame.attrs["drops_summary"]`).

### Outlier list (the only two realism caveats — both calibration, not corruption)

1. **`prob_profit == 1.000`** — 1 candidate at `as_of=2022-06-16`. A forecast of *certain* profit is
   unrealistic for any short put; it arises when the empirical forward distribution contains zero loss
   scenarios. This is the documented top-bin over-confidence and is examined quantitatively in **W3**.
2. **Thin `prob_profit` samples (`n_scenarios < 30`)** — strongly regime/as_of dependent:
   **346/358 (97%)** at calm 2021-06-15, but only 1, 0, 14, 1 at the other four dates. With the default
   5-year lookback + 35-day **non-overlapping** sampling, the empirical forward distribution can carry
   `< 30` points (the F4 finding, `docs/F4_TAIL_RISK_DIAGNOSTIC.md`). Per the honesty rule, any
   `prob_profit` from such a bin must be reported with its Wilson/bootstrap CI and not treated as a
   point estimate — **W3** stratifies accordingly.

Neither caveat is a non-finite/absurd-value defect (values stay in `[0,1]`); both are calibration
weaknesses, so they are *reported* here and *measured* in W3 — not pinned as defects. The verified
realism properties are pinned green in `tests/test_w2_output_realism.py`.

---

## §W3 — `prob_profit` calibration (realized vs forecast, by VIX regime)

**Driver:** `scripts/audit_prob_profit_calibration.py` → `w3_prob_profit_calibration.json`.
Reuses the canonical harness (`scripts/vnv_prob_profit_calibration.py`) verbatim — same
35-date feasible grid (2020-06 → 2026-02), same realized hold-to-expiry rule
(`S_expiry > strike − premium`), same `DATA_END = 2026-03-20` (which **pre-dates the W1
2026-03-23 OHLCV splice**, so realized prices are never contaminated by D-W1-1) — and adds
the **VIX-regime stratification** W3 asks for. **n = 9,564** candidate-outcomes.

> **Honesty / independence caveat.** Candidates are *not* independent (names recur, windows
> overlap), so each bin's `n` counts candidates, not trials, and the Wilson interval is
> **optimistically tight**. Cells with `n < 30` are flagged *not conclusive* and never quoted
> as support. §2: strictly read-only — every candidate is the authoritative ranker's output.

### Pooled calibration (all regimes, n = 9,564)

| `prob_profit` bin | n | forecast | realized | gap | Wilson 95% | conclusive |
|---|---:|---:|---:|---:|---|:--:|
| (0.5, 0.6] | 49 | 0.562 | 0.653 | **+0.091** | [0.513, 0.771] | ✓ |
| (0.6, 0.7] | 929 | 0.665 | 0.736 | **+0.071** | [0.707, 0.764] | ✓ |
| (0.7, 0.8] | 3405 | 0.749 | 0.752 | +0.003 | [0.737, 0.766] | ✓ |
| (0.8, 0.9] | 4345 | 0.836 | 0.792 | **−0.045** | [0.779, 0.803] | ✓ |
| (0.9, 0.95] | 767 | 0.915 | 0.818 | **−0.097** | [0.789, 0.843] | ✓ |
| (0.95, 1.0] | 68 | 0.962 | 0.824 | **−0.138** | [0.716, 0.896] | ✓ |

The miscalibration is **monotone**: the engine is *under*-confident in the low/mid bins
(realized exceeds forecast) and *over*-confident in the top two bins — the top bin forecasts
0.962 but realizes 0.824 (**−13.8 pp**, conclusive). This is the documented top-bin
over-confidence, confirmed.

### Top bin (`prob_profit > 0.90`) by VIX regime *at entry* — the headline

| VIX regime at entry | n | forecast | realized | gap | Wilson 95% | conclusive |
|---|---:|---:|---:|---:|---|:--:|
| **calm (<20)** | 150 | 0.925 | 0.767 | **−0.158** | [0.693, 0.827] | ✓ |
| **elevated (20–30)** | 420 | 0.921 | 0.764 | **−0.157** | [0.721, 0.802] | ✓ |
| **crisis (≥30)** | 108 | 0.927 | **0.935** | **+0.008** | [0.872, 0.968] | ✓ |

**The top-bin over-confidence is conclusive and ~equal in calm and elevated-vol entries
(≈ −16 pp) but VANISHES in crisis entries** — at VIX ≥ 30 the top bin is essentially
well-calibrated (realized 0.935 vs forecast 0.927, +0.008). In fact **every** bin in the
crisis-entry stratum is *under*-confident (e.g. (0.8, 0.9]: realized 0.975 vs 0.855, +0.119;
(0.7, 0.8]: 0.967 vs 0.773, +0.194). Mechanism: a crisis entry carries very high IV → fat
premium → wide breakeven, so the put is genuinely safer than the recent-vol-fit forward
distribution credits; a calm/elevated entry's forward distribution is fit on calm history and
under-prices the chance of a vol spike during the hold, so top-bin picks get blindsided.

### Is the over-confidence regime-dependent? — Yes, and *opposite* to the documented prior

The prior studies that motivate **R11/R11b** report *realized ~0.57 vs ~0.96 forecast in
crisis*. **W3 does not reproduce a 0.57 crisis realized** — under **VIX-at-entry**
classification the crisis-entry top bin realizes **0.935**. The reconciliation is a
**regime-definition difference**, not a contradiction:

- **W3 conditions on VIX *at entry*** (what a live ranker actually knows on `as_of`).
- The **prior/R11 conditions on the regime that *follows*** an elevated-vol reading
  (CLAUDE.md §2 R11: *"the regime that follows an elevated-vol reading … ~0.57 realized vs
  ~0.96 forecast in crisis"*) — i.e. entries (typically calm/elevated *at entry*) whose
  **hold period contains** a crisis. That forward-looking subset is exactly the calm/elevated
  entries that got blindsided — consistent with W3's −16 pp calm/elevated over-confidence.

Both are true; they slice different conditionings of the same phenomenon.

### Implication for R11 / R11b thresholds (report-only — no trio edit)

R11 fires on **VIX > 25 at entry** + top-bin → size-down. Mapping that onto the W3 strata:

- **Elevated (20–30, straddling the 25 cut):** top bin over-confident **−15.7 pp** (conclusive)
  → R11's size-down on elevated-vol-at-entry top-bin picks **is supported**.
- **Crisis (≥30):** top bin **well-calibrated (+0.008)** → R11 firing here sizes down
  genuinely-safe picks (a mild false-positive / forgone-premium cost).
- **Calm (<20):** top bin **equally over-confident (−15.8 pp)** but R11 **does not fire**
  (VIX < 25) → an unaddressed over-confidence the gate misses.

So the VIX > 25-at-entry cut catches the over-confident *elevated* band but over-fires into the
well-calibrated *crisis* band and misses the *calm* band. This is decision-relevant input for
the R11a/R11b thresholds — **reported only**; the trio is untouched and this is flagged to the
Windows terminal (which owns R11b this cycle). Methodology pinned green in
`tests/test_w3_calibration.py`; the headline numbers are reproducible via the driver.

---

## §W4 — Risk-free-rate + point-in-time correctness

**Driver:** `scripts/audit_risk_free_pit.py` → `w4_risk_free_pit.json`.

### Risk-free-rate "spurious 5%" defect — **REFUTED on current data (latent, not active)**

`get_current_risk_free_rate` (`engine/data_integration.py`) returns `fallback` (the EV-path
caller `wheel_runner.py:4064` passes `fallback=0.05`) when `as_of` precedes treasury coverage.
The committed `treasury_yields.csv` now covers **rate_3m from 1994-01-03 → 2026-06-05**, which
**precedes the OHLCV start (2018-01-02)**, so the fallback path is **unreachable for any
feasible `as_of`**. Served rates are the real PIT decimal each day:

| as_of | served rate_3m (decimal) | regime |
|---|---:|---|
| 2018-06-01 | 0.01905 | normalization |
| 2020-03-15 | 0.00244 | COVID ZIRP |
| 2021-05-01 | 0.000025 | ZIRP trough |
| 2024-01-02 | 0.05363 | hiking cycle |
| 1990-01-01 (pre-coverage) | **0.05** | fallback fires (no feasible OHLCV here) |

The spurious 0.05 only appears for an `as_of` before 1994 — outside any feasible window.
**Verdict: the historical defect is resolved by the treasury back-extension; it is now latent.**

### Latent EV impact (quantified, own-driver shim — no trio edit)

To quantify what the defect *would* cost if coverage ever regressed, the driver ranks
`as_of=2021-06-15` (real rate **0.000127**) twice: once with the real PIT rate, once with both
rate sources **monkeypatched to 0.05** (in this driver's process only — never the trio), and
diffs `ev_dollars` per common ticker (n=90):

| metric | ev_dollars(forced 5%) − ev_dollars(real ~0%) |
|---|---:|
| mean | **−15.20** |
| median | −10.37 |
| max (least negative) | −1.48 |

Forcing 5% **lowers** `ev_dollars` by ~$10–15/contract (the lower BSM put premium at higher `r`
dominates any collateral-carry credit). So had the spurious 5% been live in the ZIRP era, a
2020–2021 backtest would have **understated** EV by ~$15/contract — material against typical
`ev_dollars` of $50–300. (Moot today; quantifies the latent risk.)

### Point-in-time IV — **CONFIRMED**

The ranker resolves ATM IV via `_resolve_pit_atm_iv` → `get_iv_history(ticker, end_date=as_of)`
(the S23 F3 / #378 fix), not a present-day snapshot. Verified:

- **No lookahead:** for every sampled (ticker, as_of), the served IV history's max row date is
  `≤ as_of` (0 violations).
- **Moves with as_of:** the resolved IV differs across two as_of dates and matches each date's
  PIT value **exactly** — e.g. AAPL 0.2129 (2021-06-15) vs 0.2261 (2024-01-16); MSFT 0.1819 vs
  0.2554; NVDA likewise. A frozen-snapshot regression would show identical IV across dates — it
  does not. Pinned green in `tests/test_w4_risk_free_pit.py`.

---

<!-- W5 section appended by its PR -->
