# Heavy-verify 2026-06-29 — Independent re-verification of #436 W1–W2 (data-wiring + output realism)

> **Report-only. Engine / decision-trio untouched.** This is a *fresh,
> independent reproduction* of the already-merged #436 W1 (data-wiring
> accuracy, PR #440) and W2 (engine output realism, PR #441) findings, run on
> current `main` (`8c33ff6`, incl. #459) from drivers written from scratch
> against the `MarketDataConnector` public API — **not** a re-read of the
> 2026-06-27 doc and **not** a reuse of its drivers. Goal: confirm the headline
> findings still hold and surface any drift.

| | |
|---|---|
| **Provider / data path** | `SWE_DATA_PROVIDER=bloomberg` → `MarketDataConnector` (committed `data/bloomberg/*.csv`). Logged at every driver start. |
| **Branch** | `claude/mac-data-wiring-reverify-2026-06-29` (off `main` `8c33ff6`) |
| **Drivers** | `scripts/audit_w1_data_wiring_reverify.py`, `scripts/audit_w2_output_realism_reverify.py` |
| **Artifacts** | `docs/verification_artifacts/data_wiring_reverify_2026-06-29/*.json` |
| **Determinism** | Engine is seeded (`forward_distribution.py` seed=42; `ev_engine.py` per-trade blake2b seed) and **no data CSV has merged since the 2026-06-27 campaign**, so a faithful re-run is expected to reproduce the published counts **exactly** — the strongest form of reproduction. |
| **Headline** | **W1 + W2 reproduce byte-exactly.** The engine is trustworthy on its committed data; the single committed-data defect (2026-03-23 OHLCV split-scale splice) is re-confirmed present (fix held in PR #455). Two *minor* net-new observations surfaced (below). |

## 0. Bring-up (re-confirmed green)

```
provider=bloomberg  connector=MarketDataConnector
5-ticker smoke (AAPL/MSFT/JPM/XOM/UNH): finite ev_dollars/iv/premium on every
returned row (JPM event-dropped — expected); SMOKE TEST: HEALTHY
```

---

## 1. W1 — Data-wiring accuracy (per-source reproduction)

Independent sweep over the **full 511-ticker universe** via `get_ohlcv` /
`get_iv_history` / `get_corporate_actions` / `get_risk_free_rate` /
`get_dividends` / `get_fundamentals` / `get_credit_risk`. All counts are
deterministic (no sampling).

| Source | Re-verify result | Published W1 | Match |
|---|---|---|---|
| **OHLCV** | 511 tickers, **1,014,920** rows, 2018-01-02→2026-06-04; **0** non-positive close, **0** OHLC-invariant violations, **0** non-monotonic-date series | same; "0 invariant violations; monotonic" | ✅ |
| **OHLCV split-scale sweep** | exactly **6** names with a >2× single-day close move: **BKNG, CVNA** (splice artifacts) + **GL, OXY, PCG, TRGP** (real crashes) | same 6 | ✅ |
| **vol_iv — implied IV** | `hist_put_imp_vol`+`hist_call_imp_vol` = **2,062,702** served finite cells; **0** outside `(3.0, 10000]`; **0** `134217.7` sentinel leaks | "2,062,702 served cells; 0 violations; 100% clean" | ✅ (byte-exact) |
| **Treasury / RFR** | `rate_3m` raw span **1994-01-03→2026-06-05** (8,458 rows); served decimal (COVID-ZIRP 0.0015, hiking-2024 0.0536); precedes OHLCV start ⇒ spurious-5% fallback unreachable | 1994→2026; decimal; fallback unreachable | ✅ |
| **Dividends** | AAPL 91 rows, `ex_date` monotonic, **1** negative `dividend_amount` (low-severity: ex-div lockout only) | "1 negative dividend on AAPL" | ✅ |
| **Corporate actions** | AAPL 6 disruptive (`Discontinued`/`Stock Split`), `effective_date` monotonic | PASS | ✅ |
| **Fundamentals / Credit risk** | dicts served with the documented key sets (13 / 4 keys) | PASS | ✅ |

### 1.1 Defect re-confirmed — D-W1-1 OHLCV split-scale splice @ 2026-03-23

The splice is **still present on `main`** (correct — #439's fix, PR #455, is held
for the Windows snapshot re-baseline). Re-derived from the served close:

| Ticker | Disc. date | Disc. return | Implied boundary factor | Authoritative split (`get_corporate_actions`) | Split eff. date | Splice before eff. date? |
|---|---|---|---|---|---|---|
| **BKNG** | 2026-03-23 | **−95.90%** | 24.37× | **25.0** | 2026-04-06 | **yes (+14d)** |
| **CVNA** | 2026-03-23 | **−78.85%** | 4.73× | **5.0** | 2026-05-08 | **yes (+46d)** |

**Corp-action cross-check (this is the W1-mandated check, done rigorously).**
The discontinuity sits at the **2026-03-23 data-pull boundary**, which *precedes*
both split **effective dates** (2026-04-06 / 2026-05-08). A true split adjustment
would land *on* the effective date — so the break is a **pull-boundary splice
artifact** (older un-split history + recent split-adjusted slice), exactly the
#439 diagnosis. The O/H/L/C scale uniformly across the boundary (low CV) →
mechanical rescale, not a real crash.

**Reconciliation — the precise factor is 5.0, not "4.7".** The original W1
headline quoted CVNA "÷4.7" — that is the *boundary-measured* ratio (4.73),
which is noisy because the one-day jump folds a real market move into the split.
The **authoritative ratio from `get_corporate_actions` is exactly 5.0** (BKNG
25.0), matching PR #455's `÷5`. This re-verify pins the corp-action ground truth
so the diagnosis rests on the split record, not the measured jump.

> **No new issue filed.** D-W1-1 is already tracked (#439, fix in PR #455) and
> the sibling spinoff-splice is filed (#454). No trio fix is implied — data-layer
> regeneration only.

### 1.2 NFLX ~10× — REFUTED (reproduced)

`get_ohlcv("NFLX")`: **0** >2× single-day jumps. Served 2018 first close
**$20.107** (= real ~$201 ÷10 for the 2025-11-17 10:1 split). The series is
uniformly back-adjusted across all three splits (2:1 2004, 7:1 2015, 10:1
2025-11-17); strikes/premiums/spot scale together, so `prob_profit`/IV/returns
are unaffected. **Not a defect.**

### 1.3 Net-new minor observations (independent pass)

Two small things the published "0 NaN / 100% clean" framing glossed — both
**low-severity, no engine impact, no trio fix**:

1. **OHLCV NaN-price rows: BIIB & TPL, 2 rows each** — isolated days with
   `volume` present but `open/high/low/close` all NaN (BIIB 2020-11-06,
   2023-06-09; TPL 2019-05-16, 2019-07-09). 4 rows out of ~1.01M (0.0004%);
   trading-halt / no-print bars the connector serves as-is. The EV path consumes
   `close` and these NaN bars drop out of the return series; impact is nil.
   Worth a one-line awareness note, not a fix.
2. **Realized-vol columns are correctly *not* IV-gated.** The `(3.0, 10000]`
   gate applies to **implied** vol only; the realized-vol columns
   (`volatility_30/60/90/260d`) legitimately serve values below 3% for calm
   names (157 sub-3 cells across the universe; range [1.945, 328.46]; **0**
   negatives). This is correct scoping (realized vol *can* be <3%), confirmed
   and now pinned (§3). The published "2,062,702 cells" was implied-only by
   design.

---

## 2. W2 — Engine output realism (per-regime reproduction)

`rank_candidates_by_ev(as_of=…, top_n=600, min_ev_dollars=-1e9,
include_diagnostic_fields=True)` over the full universe at the same 5 regime
dates. The served IV band is gated at **both** ends — `(3.0, 10000]` percent
(the connector gate); the high-end gate closes a finite-but-absurd-IV gap an
adversarial review flagged (observed ceilings 0.46–2.74 decimal, far under the
100.0 bound). Greeks are recomputed via the canonical
`engine.option_pricer.black_scholes_all_greeks` (the ranker emits no Greek
columns) and checked against `docs/GREEKS_UNIT_CONTRACT.md`; because the put
invariants (delta∈[-1,0], γ≥0, vega≥0) are mathematically guaranteed for valid
inputs, this functions as a **degenerate-input detector + invariant check** on
the served `(spot, strike, dte, iv, r)` tuple, not a test of engine-*emitted*
Greeks (the ranker emits none).

| Regime (as_of) | n | non-finite | Greek viol. | prob∉[0,1] | iv≤3% | prem/spot median | iv ceiling | thin n<30 |
|---|---|---|---|---|---|---|---|---|
| calm 2021-06-15 | **358** | 0 | 0 | 0 | 0 | 1.07% | 0.55 | 346 |
| calm 2024-01-16 | **71** | 0 | 0 | 0 | 0 | 1.09% | 0.46 | 1 |
| crisis 2020-03-23 | **191** | 0 | 0 | 0 | 0 | **3.70%** | **2.74** | 0 |
| bear 2022-06-16 | **329** | 0 | 0 | 0 | 0 | 1.94% | 1.41 | 14 |
| bear 2022-10-14 | **37** | 0 | 0 | 0 | 0 | 1.79% | 0.92 | 1 |
| **Total** | **986** | **0** | **0** | **0** | **0** | — | — | — |

Every cell matches the published W2 (n=986; per-regime 358/71/191/329/37;
prem/spot 1.07/1.09/3.70/1.94/1.79%; iv-ceiling 0.55/0.46/2.74/1.41/0.92).
**Verdict: REALISTIC.**

- **Magnitude scaling is monotone calm < bear < crisis** — prem/spot median
  1.08% → 1.87% → 3.70%; iv-ceiling 0.51 → 1.17 → 2.74. Premiums richen and IV
  ceilings rise with stress, as expected.
- **Outliers (reproduced, in-bounds, not corruption):**
  - `prob_profit == 1.000` on **1** candidate (**AMCR**, 2022-06-16) — empirical
    forward distribution contained zero loss scenarios; the documented top-bin
    over-confidence (W3 territory, not a finiteness bug).
  - **thin `n_scenarios < 30`** prevalent in calm 2021 (346/358 = 97%) — the
    5-yr lookback + 35-day non-overlapping sampling yields <30 points for most
    calm-regime candidates. **Honesty caveat:** per-bin calibration at this n is
    not assertable; this is exactly the F4 thin-window the merged W3 study
    (PR #443) quantified with Wilson CIs. Re-verify makes no calibration claim.

---

## 3. Net-new pins (`tests/test_w1w2_reverify.py`)

Two **durable** properties (stay green *after* PR #455 lands) not covered by the
existing `tests/test_w1_data_wiring.py` / `tests/test_w2_output_realism.py`:

1. **`test_corp_action_split_ground_truth`** — `get_corporate_actions` serves
   BKNG **25.0** (eff 2026-04-06), CVNA **5.0** (eff 2026-05-08), NFLX **10.0**
   (eff 2025-11-17). Hand-verified ground truth; pins the *root-cause data* the
   existing split-scale `xfail` only describes in prose.
2. **`test_split_effective_dates_postdate_the_2026_03_23_splice`** — diagnosis
   pin: both split eff dates are strictly after the 2026-03-23 boundary
   (corp-action dates are unaffected by the OHLCV regeneration, so this survives
   the fix).
3. **`test_iv_gate_scoped_to_implied_not_realized`** (EA, HOLX) — implied IV min
   > 3.0 (gated) while realized vol min < 3.0 (not floored), 0 negatives. Pins
   the deliberate gate scoping from §1.3.

**RED-proven:** mutating the expectations (asserting the boundary ratio 24.375
instead of the true 25.0; claiming realized vol is floored >3.0) makes the pins
fail — they have teeth, not signature-only.

---

## 3b. Adversarial verification — **CONFIRMS**

Three independent skeptic agents (a driver-methodology audit + an independent W1
re-derivation + an independent W2 regime re-run, written without touching these
drivers) tried to refute the reproduction. **Verdict: CONFIRMS** — all numbers
independently reproduced (BKNG/CVNA → splice; GL/OXY/PCG/TRGP → real crash with
O/H/L/C CV 0.07–0.39 vs splice CV ≤0.021; NFLX refuted; corp-action ground
truth; EA/HOLX IV mins; W2 n=71, 0 Greek violations, median 1.085%, first-row
put delta −0.250). **No verdict-flipping error found.** Four minor, non-flipping
scope/labeling observations were raised and resolved:

| Observation | Resolution |
|---|---|
| W2 verdict gated low-IV but not high-IV → a finite-but-absurd served IV could pass | **Fixed** — added the `(…, 10000]` percent ceiling gate to the W2 verdict (`iv_above_ceiling_total`; 0 across all regimes). |
| W1 split sweep flags only **>2×** moves → a splice on a (1.5, 2.0)-ratio split would be missed | **Documented + verified non-issue** — every (1.5,2.0)-ratio split in the data is 1983–2017 (uniformly back-adjusted long ago); the only post-2026-03-23 splits are BKNG 25 / CVNA 5 / CRWD 4 / KLAC 10 (all >2×), and CRWD/KLAC carry **no** discontinuity (correctly absent from the suspects). No real splice is masked. |
| W1 implied-IV / sentinel checks read *post-gate* values → can't detect a gate-*tightening* regression | **Accepted (non-flipping)** — they still catch a broken/unapplied gate (an out-of-band finite cell is flagged), and `test_iv_gate_scoped_to_implied_not_realized`'s `realized_min < 3.0` assertion carries independent scoping signal. |
| Greek check re-derives canonical BSM Greeks (ranker emits none) → narrower than "Greek-contract violations" implies | **Clarified** — relabeled in §2 as a degenerate-input detector + invariant check on served inputs. |

## 4. Deliverables

- This doc (every claim carries its data path; deterministic counts, no
  sampling — reproduction is exact).
- `scripts/audit_w1_data_wiring_reverify.py`, `scripts/audit_w2_output_realism_reverify.py`
  (persist JSON before pretty-printing; `PYTHONIOENCODING=utf-8`).
- `docs/verification_artifacts/data_wiring_reverify_2026-06-29/` — `w1_*.json`,
  `w2_*.json`, summaries.
- `tests/test_w1w2_reverify.py` — 6 passing pins (RED-proven).
- **No new issues** (D-W1-1 → #439/#455; sibling → #454; both pre-existing).
  **No trio touch.**

## 5. Bottom line

The merged #436 W1–W2 reliability findings **reproduce exactly** on current
`main` from independent drivers: the committed data is wired faithfully, the
engine's outputs are finite/realistic/regime-scaled, and the only committed-data
defect is the already-tracked 2026-03-23 split-scale splice (fix held in #455).
The independent pass additionally tightened the CVNA factor to the authoritative
**5.0** and pinned the corp-action ground truth + IV-gate scoping.
