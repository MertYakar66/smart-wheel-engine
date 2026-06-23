# PHASE 2 — Skew-surface execution spec (the prize)

_Compiled 2026-06-23 · **PLAN ONLY — docs, no code.** A turnkey, file:line-accurate
implementation spec for wiring the **moneyness IV skew surface** that `docs/WIRING_CAMPAIGN.md`
Phase 2 schedules as **supervised, EV-moving, PANEL, re-baseline-coupled**. Every line
reference was verified against **`origin/main` @ `21e489d`** (2026-06-23, post #416/#417/#418)
and adversarially re-checked. This doc changes no engine code._

**Read with:** `docs/WIRING_CAMPAIGN.md` Phase 2 (lines 182–200 — the surface row + honest
limits), `docs/PHASE1_E_TRIO_EXECUTION_SPEC.md` (#378 must precede this), `docs/DATA_TEST_AUDIT_2026-06-09.md`
(C4 / W28 — the `edge_vs_fair`-stays-0 fact), `CLAUDE.md` §2/§3, `data/broad_pull_loaders.py`
(the dormant surface loader this wires).

> **The data is already banked and loadable.** `#417` committed the surface at
> `data/bloomberg/broad_pull/iv_surface/sp500_iv_surface.csv.gz` (1,944,699 rows · 509 names ·
> 2010-01-04 → 2026-06-17) with a tested, **dormant** `BroadPullLoader`. Phase 2 is the
> *consumption* step — nothing here re-pulls or re-bakes data.

---

## 0. Standing rules (do not relax)

- **Phase 2 is one PANEL step, not the trio's CEREMONY.** Held for review; **§2 panel**;
  no verdict logic; **no `EVEngine.evaluate` bypass**; **downgrade-only** contract intact;
  the dealer-multiplier clamp `[0.70,1.05]` untouched; matching characterization tests
  **flipped, not deleted**. **No autonomous land.**
- **EV-moving → re-baseline-coupled.** It lands **before** the single S27/S32/S34/S35 re-pin
  (Phase R), which absorbs its `ev_raw` shift in one pass. **Do NOT trigger a second
  re-baseline** (don't pay the ~4–6.5h tax twice — `WIRING_CAMPAIGN.md` lines 33–37).
- **⚠️ `#378` (IV-staleness gate) must land first.** Phase 2 makes the served IV load-bearing
  for the BSM fair value; `#378` guards the IV↔spot staleness gap before the surface IV is
  consumed (`PHASE1_E_TRIO_EXECUTION_SPEC.md` §3 + `WIRING_CAMPAIGN.md` §0A note).
- **🔒 The lane-claim CI gate is SILENT here.** `DECISION_LAYER_FILES` is the trio only —
  `engine/ev_engine.py` · `engine/wheel_runner.py` · `engine/candidate_dossier.py`
  (`scripts/check_lane_claim.py:80-83`). `option_pricer.py` / `skew_dynamics.py` /
  `data_connector.py` are **absent**, so CI passes **without** a lane-claim block. That is
  expected for PANEL — but it means **the human §2 panel is the sole governance gate.** Do not
  read green CI as governance sign-off.

### Three scope corrections vs the campaign doc (carry these in — do not re-derive)

The `WIRING_CAMPAIGN.md` Phase-2 row (line 191) names "moneyness-aware IV; **vanna/charm/volga**"
and `skew_dynamics` "risk-reversals, **butterflies**". Adversarial code-reading found three of
those are mis-scoped:

- **(C1) `vanna/charm/volga` already exist.** `engine/option_pricer.py` computes them on the
  scalar path — `vanna :524`, `charm :530-536`, `volga :540`, assigned `:557-559` — and
  `dealer_positioning` already consumes them. **They are not a Phase-2 deliverable.** The only
  gap is the **vectorized** path: `vectorized_bs_all_greeks` (`option_pricer.py:1181`, second-
  order block ~`:1262`) omits them. Add them there **only if** a batch surface-aware greek
  consumer is introduced — otherwise out of scope.
- **(C2) Butterflies do NOT exist in `skew_dynamics`.** The module ends at `:267` with only the
  25Δ `risk_reversal` (`skew_dynamics.py:167`). A butterfly lives only in the **dormant SVI**
  module `engine/volatility_surface.py` (do not confuse the two). The campaign's "butterflies"
  is **new code** — scope it explicitly in the PR or defer it; a coverage agent must not report
  it as wired.
- **(C3) The surface has NO true 25Δ column.** It is a fixed **5×5** moneyness grid
  `{90,95,100,105,110}` (wings `{80,120}` **empty**); 25-delta points sit nearer ~85/115. The
  connector must map `{90}`→put leg / `{110}`→call leg as a **labeled moneyness proxy** (or
  convert moneyness→delta). What `skew_slope` then returns is a **90/110-moneyness risk
  reversal**, *not* a literal 25Δ RR — **name it honestly** in the output column and tests.

---

## 1. The connector surface accessor (the foundation)

### Finding
`engine/data_connector.py` has a tight "Volatility & IV" accessor family but **no surface
accessor**, and its CSV loader **cannot read the broad-pull `.gz`**.

| Surface | Location | Current behaviour |
|---|---|---|
| IV accessor template | `engine/data_connector.py:479-505` | `get_iv_history(self, ticker, start_date=None, end_date=None)` → `self._load("vol_iv")` → `_filter_ticker :360` / `_filter_dates :343` → DataFrame, `DatetimeIndex` named `date` |
| PIT emulation helper | `engine/data_connector.py:507-526` | `_iv_series` derives a trailing window from `as_of - lookback*1.6 d` |
| Derived scalars | `:528` `get_iv_rank`, `:547` `get_iv_percentile`, `:563` `get_vol_risk_premium` | all PIT-emulate via `end_date=as_of` |
| Loader limit | `_load` / `_FILES` (`data_connector.py:77`) | reads **plain CSVs under `data/bloomberg/` only** — cannot reach `broad_pull/iv_surface/…csv.gz` |
| Surface already loadable | `data/broad_pull_loaders.py:352` | `series(name, ticker, as_of)` returns the **PIT-filtered, ticker-normalized** per-name surface time series (dormant) |
| ATM single source today | `engine/data_connector.py:900` | `get_fundamentals` serves `"implied_vol_atm": r.get("30day_impvol_100.0%mny_df")` — the **current ATM IV** |

### Fix
Add a **read-only** surface accessor that **delegates to `BroadPullLoader`** (the connector's
own `_load`/`_FILES` path cannot read the gz/subdir):
- Insert `get_iv_surface(self, ticker, start_date=None, end_date=None)` (+ an `as_of` PIT
  variant) **after `get_vol_risk_premium` (ends ~`:580`)**, mirroring `get_iv_history`'s
  shape: returns a DataFrame with the 25 `iv_{30,60,90,180,365}d_{90,95,100,105,110}` columns
  and a `DatetimeIndex` named `date`.
- Delegate the read to `BroadPullLoader.series("iv_surface", ticker, as_of)`
  (`broad_pull_loaders.py:352`) — which already PIT-clips `≤ as_of` and resolves the
  Bloomberg-format raw ticker (`"A UN"`) via `ticker_normalized`. **Do not** match on the raw
  `ticker` column or every name silently misses.
- **Honest 5×5 (C3):** never fabricate `{80,120}` wings; the accessor returns only the
  populated `{90,95,100,105,110}` columns and **raises/skips loudly on missing data** (the D9
  "never a flat-IV stub" contract).
- **Precision note:** `BroadPullLoader._downcast_floats` (`broad_pull_loaders.py:269`) casts to
  **float32**, vs the connector's float64 `vol_iv` path. Surface IVs are float32 — acceptable
  for IV inputs, but document it so a greek consumer never compares cell-by-cell against float64.

### §2 · Tests
- **§2 role:** evaluate-input-correctness (read-only input serving). **PANEL.** Not EV-moving
  *by itself* — it moves EV only once §2/§3 consume it.
- **Add:**
  - `tests/test_data_connector.py::test_get_iv_surface_returns_5x5_columns` — returns the 25
    `iv_*` columns + `DatetimeIndex 'date'`, mirroring `get_iv_history`; raises/skips loudly on
    missing data (no flat-IV stub).
  - `tests/test_data_connector.py::test_get_iv_surface_refuses_empty_wings_80_120` — the
    accessor never fabricates `{80,120}` (5×5 honesty).
- **Keep (loader-side guardrails, don't loosen):**
  - `tests/test_broad_pull_loaders.py::test_series_pit_filter` (`:332`) + the normalized-ticker
    real-data test — confirm `series("iv_surface", …)` PIT-clips and resolves `"A UN"`.

---

## 2. `skew_dynamics` reactivation (the sizing path)

### Finding
`engine/skew_dynamics.py` is a pure-numpy library; on the **Bloomberg CSV path it is dormant**
because its only live caller is gated on a chain it never receives.

| Surface | Location | Current behaviour |
|---|---|---|
| 25Δ skew function | `engine/skew_dynamics.py:151` | `skew_slope(iv_25d_put, iv_atm, iv_25d_call)` → dict; `risk_reversal = iv_25d_call − iv_25d_put` `:167`; `slope = (put−call)/max(atm,1e-6)` `:168` |
| Term-structure consumer | `engine/skew_dynamics.py:57` (`NelsonSiegelTermStructure`, `.fit :64`), `:229` (`ivs_dislocation_score`) | take 1-D `(tenors_years, ATM-IVs)` arrays — the **100%MNY column across tenors**, not the moneyness axis |
| Only live engine caller | `engine/wheel_runner.py:1540` (import), `:1557` (call) | gated `:1538` `if use_skew_dynamics and chain_df is not None …` (default `use_skew_dynamics=True :812`); legs picked from per-strike chain deltas |
| What it drives | `engine/wheel_runner.py:1561` | `skew_mult = clip(1.0 − 0.5·slope, 0.85, 1.08)` — a **sizing** multiplier (distinct from the dealer clamp `[0.70,1.05]`), feeds the regime mult |
| Dormancy | — | the Bloomberg CSV path supplies **no `chain_df`** → block never fires. `NelsonSiegel` / `skew_momentum` / `ivs_dislocation_score` have **zero** engine callers |

### Fix
Add a **surface-fed alternative branch** so the dormant block reactivates without a chain:
- At `wheel_runner.py:1538`, add an `elif`/fallback: when `chain_df` is absent but the new
  `get_iv_surface` accessor returns data for `(ticker, as_of)`, pull the three legs from the
  surface at the trade's tenor — **`{90}`→put, `{100}`→atm, `{110}`→call (C3 proxy)** — and call
  `skew_slope(iv_90, iv_100, iv_110)`.
- **Name the output honestly:** the emitted `risk_reversal` / `skew_slope` columns are a
  **90/110-moneyness** RR, not a literal 25Δ RR (C3). Keep the existing `skew_mult` clamp
  `[0.85,1.08]` — this stays **advisory-sizing**, never an `ev_raw` multiplier.
- **NaN guard:** `skew_slope:168` guards a zero ATM but **not** NaN/inf. Replicate
  `wheel_runner`'s `0 < v ≤ 3.0` pre-filter (the live block already does this at the chain
  path) before calling, or the surface path emits a silent NaN with no flag.
- **Term structure (optional, separate):** feeding the ATM (100%MNY) column across the five
  tenors into `NelsonSiegelTermStructure.fit` / `ivs_dislocation_score` reactivates regime +
  dislocation scoring — **but those have no consumer today.** Wiring the accessor alone does
  **not** reactivate them; a new caller (signal context / `candidate_dossier`) is its own
  scoped step. Flag, don't silently assume.

### §2 · Tests
- **§2 role:** advisory-sizing (the `skew_mult` mechanism). **PANEL.** EV-moving via the size
  multiplier on the **S34** portfolio-context snapshot.
- **Add:**
  - `tests/test_skew_dynamics_invariants.py::test_skew_slope_from_5x5_surface_proxy` — pin the
    90/110 RR on 5-point inputs; assert **no `IndexError`/`KeyError`** when wings `{80,120}` are
    absent (C3); assert the column is labeled as a moneyness proxy, not "25Δ".
  - `tests/test_wheel_runner_*.py::test_skew_reactivates_on_surface_without_chain` — with the
    accessor wired, `skew_slope` fires on the Bloomberg path (no `chain_df`); `skew_source`
    flips off "unavailable"; NaN/inf surface IV is pre-filtered, not emitted silently.
- **Keep:** `tests/test_skew_dynamics_invariants.py` (NS fail-fast, degenerate `n<2` fits,
  `skew_momentum` empty→NaN, dislocation `[-1,1]`) + `tests/test_quant_upgrades.py` (the 89%
  pure-math regression baseline) — unchanged.

---

## 3. `option_pricer` / EV fair value (the EV-moving core)

### Finding
`engine/option_pricer.py` is a pure scalar BSM/Merton library; **IV enters as one flat scalar**
and on the EV path that scalar is the **ATM** IV.

| Surface | Location | Current behaviour |
|---|---|---|
| BSM core | `engine/option_pricer.py:126` (`black_scholes_price`, `sigma` param) | single scalar `sigma`; `d1/d2 :170-176` |
| EV fair value | `engine/ev_engine.py:371-381` | `fair = black_scholes_price(… sigma=trade.iv …)` `:376`; `edge_vs_fair = (premium − fair)·mult` `:380-381` (import `:88`) |
| IV field | `engine/ev_engine.py:116` | `EVTrade.iv: float` — populated upstream from the **ATM** `implied_vol_atm` (`data_connector.py:900`) |
| Distribution vol | `engine/ev_engine.py:671` | `sigma = max(trade.iv, 1e-4)` — the physical lognormal vol, **same `trade.iv`** |
| `edge_vs_fair` (no-data/lockout) | `engine/ev_engine.py:321` | `edge_vs_fair=0.0` in the event-lockout/blocked branch |
| Greeks (C1) | `engine/option_pricer.py:524/530-536/540` | `vanna/charm/volga` **already exist** (scalar path); **absent** from `vectorized_bs_all_greeks` (`:1181`, ~`:1262`) |

### Fix
Make the BSM IV **moneyness-aware** at the seam — the **caller resolves a per-strike sigma**;
the pricer signature is unchanged:
- **Highest-leverage change — `ev_engine.py:376`:** replace `sigma=trade.iv` with an IV resolved
  from the surface at the candidate's **moneyness `K/S`** and **DTE tenor** (interpolate within
  the populated `{90,100,110}` × `{30,60,90,180,365}d` grid; clamp honestly, never fabricate
  `{80,120}` wings — C3). The ATM `{100}` slice **must equal** the legacy `trade.iv`.
- **Distribution-vol decision (`ev_engine.py:671`) — deliberate:** decide whether the physical
  distribution uses **strike IV** (skew) or **ATM IV** (term structure). Splitting fair-vol
  (strike) from distribution-vol (ATM) **desynchronizes** the risk-neutral and physical legs in
  a way that **moves EV** — make this an explicit §2-panel decision, not an accident.
- **`edge_vs_fair` stays structurally 0 (C4 / W28):** on the synthetic path the premium
  (`wheel_runner` synthetic-premium sites) and `fair` (`ev_engine.py:376`) derive from the
  **same** `black_scholes_price` with the **same** sigma, so a sharper surface IV improves the
  price **level** but creates **no** VRP/edge. Reviving VRP needs a market-mid premium producer
  the connector lacks. **Do not let skew wiring be reported as turning VRP live.**
- **IV gate lives upstream:** non-finite `sigma` is deliberately **not** raised in
  `option_pricer` (`_validate_inputs` note `:59-67`) — a NaN surface IV prices to NaN and is
  blocked downstream by R1a, not at the pricer. The validity gate must live in the **surface/
  data layer** (§1) — and `#378`'s staleness gate must precede this consume.
- **Greeks (C1):** scalar `vanna/charm/volga` already exist; **only** add them to
  `vectorized_bs_all_greeks` (`:1181`) **if** a batch surface-aware greek consumer is introduced.

### §2 · Tests
- **§2 role:** evaluate-input-correctness (the primary BSM-IV seam feeding `EVEngine.evaluate`).
  **PANEL.** **EV-moving** — folds into Phase R.
- **Add:**
  - `tests/test_ev_engine_*.py::test_fair_uses_surface_iv_at_moneyness` — fair-value sigma is
    resolved from the surface at the candidate `K/S` + DTE tenor, **and** the `{100}` ATM slice
    equals the legacy `trade.iv` (single source of truth).
- **Keep (guard the honest limit — do not loosen):**
  - `tests/test_ev_engine_*.py::test_edge_vs_fair_stays_zero_on_synthetic_path` — even with a
    sharper surface IV, `premium == fair` on the synthetic path ⇒ `edge_vs_fair == 0`
    (C4 / W28). Add it if no equivalent exists.
  - `tests/test_option_pricer.py::test_vectorized_all_greeks_lacks_second_order` — document the
    `vectorized_bs_all_greeks` gap so callers don't assume vanna/charm/volga exist there.

---

## 4. ATM single-source-of-truth + the `#378` dependency

The `100%MNY` column is **three potential ATM sources** that must agree:
1. the surface `iv_{tenor}d_100` (new, §1),
2. `get_fundamentals` `implied_vol_atm` ← `30day_impvol_100.0%mny_df` (`data_connector.py:900`, today's `trade.iv`),
3. the **Phase 3C** ATM IV term structure (`vol_term_rv`).

**Pin the overlap:** the surface's `{100}` slice must reconcile to `implied_vol_atm` (a test
asserting equality at the 30d tenor), or the engine carries two divergent ATM IVs. And per the
standing rule, **`#378` (IV-staleness gate) lands before this consume** — the surface frontier
(06-17) leads the legacy ATM-IV monolith, so an ungated stale IV would silently feed the new
fair value. (`PHASE1_E_TRIO_EXECUTION_SPEC.md` §3; `WIRING_CAMPAIGN.md` line 198.)

---

## 5. §2 panel checklist (the sole governance gate)

Because the lane-claim CI gate is silent here (§0), the operator panel is the only gate:

- [ ] **No bypass.** Every tradeable path still routes through `EVEngine.evaluate`
      (`CLAUDE.md` §2); the surface accessor is read-only input — it introduces no path that
      converts a non-tradeable candidate to tradeable.
- [ ] **Role = evaluate-input-correctness only.** The surface changes the **value** of the BSM
      IV fed to `black_scholes_price` (`ev_engine.py:376`) and hence `ev_raw`; it adds **no**
      verdict logic, **no** new gate, **no** new reviewer.
- [ ] **Downgrade-only intact.** No skew-derived path can upgrade/rescue a verdict; the
      `candidate_dossier` never-upgrade invariant (`candidate_dossier.py:23`) is untouched.
- [ ] **Dealer clamp untouched.** `[0.70,1.05]` still scales only final `ev_dollars`, never
      `ev_raw`; skew is **not** wired as a new `ev_raw` multiplier. (`skew_mult [0.85,1.08]`
      stays advisory-sizing — §2.)
- [ ] **`edge_vs_fair` stays 0.** Assert the synthetic-path `edge_vs_fair` is unchanged at 0
      (C4 / W28); reject any claim that skew revives VRP.
- [ ] **5×5 honesty.** No `{80,120}` wing fabrication anywhere; the 90/110 RR is labeled a
      moneyness proxy, not a literal 25Δ RR (C3).
- [ ] **ATM reconciliation.** The surface `{100}` slice equals `implied_vol_atm` and the
      Phase 3C term structure (§4); `#378` landed first.
- [ ] **Lane-claim scope.** `option_pricer`/`skew_dynamics`/`data_connector` are not in
      `DECISION_LAYER_FILES` — CI passes without a claim block; the panel is the substitute
      control. Fill the PR template's `## §2 surface` section
      (`.github/pull_request_template.md:28-30`).
- [ ] **Characterization tests flipped, not deleted;** new behaviour tests added (§§1–3).
- [ ] `pytest tests/ -m "not backtest_regression"` green; `ruff check` + `ruff format --check`
      clean; coverage ≥ 80%. `FILE_MANIFEST` / `TESTING.md` updated for new tests; worklog +
      INDEX regenerated.
- [ ] **Held for review; re-baseline-coupled** — no autonomous land; no separate re-baseline.

## 6. After it lands → Phase R (single re-baseline)

Phase 2's `ev_raw` shift is absorbed by the **one** S27/S32/S34/S35 re-pin alongside the
(E) trio and the Phase-0 data change — **not** a separate pass. Re-pin via `--update-snapshot`
(`backtests/regression/s27_ivpit_24t_100k.py:84`, `s32_friction_24t_1m.py:84`,
`s34_universe_100t_1m.py:80`, `s35_oos_24t_100k.py:78`); confirm the fingerprint guard goes red
first, bump `EXPECTED_FRONTIER` + the data-test `FRONTIER`, clear S34's provisional flag;
operator-merge, no auto-update. (`docs/NEXT_DATA_SESSION_RUNBOOK.md` Phase 3/4 +
`docs/WIRING_CAMPAIGN.md` Phase R.)

## 7. Cross-references

- `docs/WIRING_CAMPAIGN.md` — Phase 2 (the surface row + honest limits) + Phase R coupling.
- `docs/PHASE1_E_TRIO_EXECUTION_SPEC.md` — the (E) trio; **`#378` precedes this**.
- `docs/DATA_TEST_AUDIT_2026-06-09.md` — C4 / W28 (`edge_vs_fair` stays 0; no VRP revival).
- `data/broad_pull_loaders.py` — the dormant `BroadPullLoader.series("iv_surface", …)` this wires.
- `CLAUDE.md` §2/§3 + the `EnginePhaseReviewer` R1–R11 rules.
