# PHASE 1 — (E) trio execution spec (#372 · #369 · #378)

_Compiled 2026-06-23 · **PLAN ONLY — docs, no code.** A turnkey, file:line-accurate
implementation spec for the three (E) trio / risk-gate fixes that `docs/WIRING_CAMPAIGN.md`
Phase 1 schedules as **supervised, EV-moving, re-baseline-coupled**. Every line reference
was verified against **`origin/main` @ 83eacdd** (2026-06-23) — not the 2026-06-09 audit,
whose numbers have drifted (e.g. `get_current_risk_free_rate` moved `:323→:300`; the ranker
`sector` column moved `:1777→:1822/2751/3379`). This doc makes the supervised session fast;
it changes no engine code._

**Read with:** `docs/NEXT_DATA_SESSION_RUNBOOK.md` (Phase 2 = the (E) trio; the re-baseline
rule), `docs/DATA_TEST_AUDIT_2026-06-09.md` (W17/W27/W36/W37 + C1–C4), `docs/WIRING_CAMPAIGN.md`
(the campaign + the #378-before-0A ordering), `CLAUDE.md` §2.

---

## 0. Standing rules (do not relax)

- **Order: `#372 → #369 → #378`** (runbook Phase 2). Each is a **separate lane-claimed trio
  PR**, **held for review**, with a **§2 panel** the operator verifies, the **downgrade-only**
  contract intact, **no `EVEngine.evaluate` bypass**, and the **matching characterization test
  flipped (not deleted)**. **No (E) fix lands autonomously.**
- **All three are EV-moving → re-baseline-coupled.** They land **before** the single
  S27/S32/S34/S35 re-pin (Phase R), which absorbs their `ev_mean` impact + the data change in
  one pass.
- **⚠️ #378 must also precede Phase 0A.** 0A advances the OHLCV/spot frontier to 06-18 while
  the legacy ATM-IV monolith (`sp500_vol_iv_full.csv`, no in-repo producer) stays ~06-04 —
  opening the ~10-day IV↔spot gap #378 guards. So land #378 **before 0A executes**, or have
  0A re-pin the served ATM IV in the same step (see `docs/WIRING_CAMPAIGN.md` §0A note).
- **CEREMONY** each (decision-trio / risk-gate). Lane-claim block names the file; §2 panel;
  held.

---

## 1. #372 (HIGH) — R9 sector cap → real GICS, not `DEFAULT_SECTOR_MAP`

### Finding (audit W17 / C2)
The R9 sector grouping — hard gate, soft-warn, **and** the ranker's `sector` column — is
driven end-to-end by the static, hand-maintained `DEFAULT_SECTOR_MAP` (**132 names, 11
sectors**), which silently buckets every off-list ticker into `"Unknown"`.

| Surface | Location | Current behaviour |
|---|---|---|
| The map | `engine/risk_manager.py:1579-1723` | `DEFAULT_SECTOR_MAP: dict[str,str]` — 132 hardcoded entries; no GICS feed |
| Grouping | `engine/risk_manager.py:1758-1760` | `get_sector` → `self.sector_map.get(symbol, "Unknown")`; map set `:1755` `sector_map or DEFAULT_SECTOR_MAP` |
| Hard gate | `engine/portfolio_risk_gates.py:343,372,384` | `check_sector_cap` builds `SectorExposureManager(sector_map=None→DEFAULT)`; `sector = manager.get_sector(symbol)` `:384`; emits `reason="sector_cap_breach"` `:403` |
| Soft-warn R9 | `engine/candidate_dossier.py:483-488,490,497` | calls `check_sector_cap(...)` with **no** sector_map; `sector = sector_result.details.get("sector","Unknown")` `:490`; `return "review","sector_cap_breach",notes` `:497` |
| Ranker column | `engine/wheel_runner.py:1822` (put), `:2751` (CC), `:3379` (strangle) | `"sector": DEFAULT_SECTOR_MAP.get(ticker,"Unknown")` (import `:34`; schema labels `:243/:292`) |

**Key low-risk fact:** the **real GICS is already in scope.** `get_fundamentals` exposes it —
`engine/data_connector.py:897` `"sector": r.get("gics_sector_name")` — and `wheel_runner`
already loads it at `:504` `analysis.sector = fundamentals.get("sector","")` (reused in the
summary `sector_allocation` `:3813-3817`). The ranker rows simply **ignore** it.

### Fix
Thread real `gics_sector_name` as the **primary** sector source, with a **counted** `"Unknown"`
fallback (never silent), so gate + soft-warn + ranker agree:
- **Ranker rows** `:1822/2751/3379`: replace `DEFAULT_SECTOR_MAP.get(ticker,"Unknown")` with
  the real GICS already loaded (`analysis.sector` / `fundamentals["sector"]`), counted-`Unknown`.
- **Gate + R9**: build a per-run sector map from `gics_sector_name` (primary) → `DEFAULT_SECTOR_MAP`
  (fallback for the ~NaN names) → counted `"Unknown"`, and pass it via the **existing**
  `sector_map=` param `check_sector_cap` already accepts (`portfolio_risk_gates.py:350`) — the
  minimal-surface path (no new gate signature). R9 (`candidate_dossier.py:483`) and the
  tracker's `open_short_put` caller pass that map.
- **Source = main `fundamentals.gics_sector_name`** (W19 already pinned the canonical 11; #373
  landed). **Do NOT** source from the broad-pull `snapshot_bdp` (single 2026-06-18 as-of →
  lookahead). Pin the `GICS_11` set; count + log the `"Unknown"` fallbacks.
- No connector change (it already serves `gics_sector_name`).

### §2 · Tests
- **§2 role:** downgrade-only (R9 soft-warn) + the hard gate; GICS data is
  evaluate-input-correctness. **CEREMONY.** EV-moving on the **S34** portfolio-context snapshot.
- **Flip (characterization → GICS-primary):**
  - `tests/test_data_integrity_bloomberg.py:494::test_r9_sector_map_ignores_pulled_gics_characterization` — today pins map-ignores-GICS; flip to assert GICS-primary resolution.
  - `tests/test_ranker_transparency.py:449::test_sector_column_uses_default_sector_map` — today asserts ranker `sector == DEFAULT_SECTOR_MAP.get(...)`; flip to assert it equals the connector GICS (`get_fundamentals(t)["sector"]`).
  - `tests/test_ranker_transparency.py::test_sector_column_present_on_all_survivor_rows` — keep present/non-empty asserts; update docstring/intent to "real GICS, counted Unknown".
  - `tests/test_portfolio_risk_gates.py:187::...test_empty_portfolio_passes` + `:228::test_unknown_sector_uses_unknown_bucket` — sector must come from the GICS source; `"Unknown"` only on a counted GICS-miss.
  - `tests/test_dossier_r9_r10_audit.py::test_r9_aggregates_multiple_held_same_sector_positions` + `::test_r9_unknown_sector_ticker_aggregates_under_unknown` — supply a real-GICS source in the fixture; breach still fires; Unknown via counted GICS-miss.
  - `tests/test_risk_manager.py::TestSectorExposureManager::test_get_sector_known/_unknown_returns_unknown` — explicit-map override stays; add a counted-GICS-miss case.
- **Keep:** `tests/test_data_integrity_bloomberg.py:484::test_fundamentals_gics_sector_is_canonical_11` (W19).
- **Flips green when landed:** `tests/test_broad_pull_wiring_xfail.py::test_r9_sector_grouping_uses_real_gics` (Phase 0B scaffold) — drop its `xfail` marker.
- **Dependencies:** W19/#373 landed; GICS already connector-served. No 0A/0B-data dependency.

---

## 2. #369 — extend the #363 IV gate to the fundamentals-fallback path

### Finding (audit W27)
The #363 connector gate cleans only the `vol_iv` IV columns; the ranker's **IV fallback** reads
**uncleaned** IV from `sp500_fundamentals.csv`, normalised only by an inline `if iv > 3.0:
iv/=100` heuristic duplicated in **four** places.

| Surface | Location | Current behaviour |
|---|---|---|
| #363 gate | `engine/data_connector.py:317-334` | `_clean_vol_iv_inplace` NULLs `_DEEP_IV_COLS` (`hist_put/call_imp_vol`) outside band `(_IV_LOW_FLOOR=3.0 :140, _DEEP_IV_SENTINEL_FLOOR=10000.0 :125]` |
| Gate scoping | `engine/data_connector.py:227-228` (`_load`), `:295-296` (`_load_assembled`) | runs **only** `if key == "vol_iv"` |
| Fundamentals serve | `engine/data_connector.py:863-901` | `get_fundamentals` → `_load("fundamentals")` (key≠vol_iv → **no gate**); returns raw `implied_vol_atm` / `volatility_30d` `:899` |
| Fallback (put) | `engine/wheel_runner.py:1124-1147` | `iv=_resolve_pit_atm_iv(...)`; if `None` → `iv_raw=fundamentals.get("implied_vol_atm")` `:1127` (then `volatility_30d` `:1129`) + inline `if iv>3.0: iv/=100` |
| Fallback (CC) | `engine/wheel_runner.py:2459-2475` (`:2462`) | mirror |
| Fallback (strangle) | `engine/wheel_runner.py:3032-3048` (`:3035`) | mirror |
| 4th heuristic copy | `engine/wheel_runner.py:198-199` | inside `_resolve_pit_atm_iv` (PIT path — already gated upstream by #363; documents the duplication) |

### Fix
Clean the **fundamentals-served** IV at the connector (in `get_fundamentals`, or the
`fundamentals` load) using the **#363 band** — NULL a sub-3.0 / >sentinel `implied_vol_atm`
before it reaches the ranker — so a corrupt value never passes raw and the inline `if iv>3.0`
heuristic stops being the **sole** normaliser (it may stay as a defensive belt). Reuse
`_clean_vol_iv_inplace`'s band (a sibling cleaner over the fundamentals IV cell). **Connector-
layer serving change feeding `EVEngine.evaluate`; no verdict logic — but it moves EV magnitudes.**

### §2 · Tests
- **§2 role:** evaluate-input-correctness. **CEREMONY** (the runbook places it in the trio lane).
- **Flip:**
  - `tests/test_data_to_engine.py:482::test_363_gate_does_not_clean_fundamentals_iv` — today asserts the fundamentals IV is **not** cleaned (a 2.0 passes through); flip to assert the sub-3.0 `implied_vol_atm` is now NULLed by the connector.
  - `tests/test_data_to_engine.py:471::test_fundamentals_fallback_iv_input_is_percent` — today asserts served IV is percent (`>3.0`); flip **only if** the fix also moves percent→decimal into the connector (assert decimal). If the fix only NULL-bands (keeps percent for valid values), this stays — decide in the §2 panel.
- **Note:** the percent→decimal heuristic is duplicated in 4 places (`:198-199`, `:1127-ish`, `:2462-ish`, `:3035-ish`); cleaning at the connector lets a follow-up collapse them. Not required for #369.

---

## 3. #378 — IV-staleness gate on `_resolve_pit_atm_iv` + rate-fallback divergence

### Finding (audit W36 / W37)
`_resolve_pit_atm_iv` takes the last IV row with **no staleness gate** (the spot path *has* a
30-day gate); and the EV-path rate accessor returns a **silent `0.05`** before treasury
coverage, diverging from the connector's NaN-on-missing contract.

| Surface | Location | Current behaviour |
|---|---|---|
| IV PIT helper (no gate) | `engine/wheel_runner.py:153,178,187,193` | `hist=get_iv_history(ticker,end_date=as_of)` `:178`; `row=hist.iloc[-1]` `:187` — no check of the IV bar's date vs `as_of` |
| Spot gate to **mirror** | `engine/wheel_runner.py:475,533-545` | `max_as_of_staleness_days:int=30` `:475`; `gap_days=(cutoff-ohlcv_pit.index.max()).days`; `if gap_days <= max_as_of_staleness_days:` use, else stale |
| Rate fallback (silent 0.05) | `engine/data_integration.py:300,323,330,338,342` | `get_current_risk_free_rate` returns `0.05` for missing file `:323` / missing tenor `:330` / empty-after-`as_of` (before coverage) `:338` / NaN `:342` |
| Rate accessor wired | `engine/wheel_runner.py:588-590` | `analysis.risk_free_rate = get_current_risk_free_rate(as_of,...)`; local `except → 0.05` `:592` |

### Fix
- **IV staleness:** in `_resolve_pit_atm_iv`, after `hist.iloc[-1]`, read the row's date
  (DatetimeIndex `.max()` or a `date` column), compute `gap_days` vs `pd.Timestamp(as_of)`,
  and **return `None`** (→ legacy fallback) when `gap_days > max_as_of_staleness_days` (30,
  mirroring `:533-545`). Thread the same 30-day constant.
- **Rate divergence:** make `get_current_risk_free_rate` return `float("nan")` (or raise) for
  missing-file / missing-tenor / before-coverage / NaN — aligning with the connector's
  `get_risk_free_rate` NaN-on-missing — and have the caller (`wheel_runner.py:588-592`) treat
  NaN as "no rate" (drop/flag, not silently price at 0.05). **Or** document the divergence as
  intentional with an explicit guard (decide in the §2 panel).

### §2 · Tests
- **§2 role:** evaluate-input-correctness. **CEREMONY.**
- **Keep (data-side regression guards, don't loosen):**
  - `tests/test_data_integrity_bloomberg.py:733::test_vol_iv_ohlcv_last_date_consistency` (W36).
- **Flip:**
  - `tests/test_data_integrity_bloomberg.py:758::test_data_integration_rate_before_coverage_divergence` (W37) — today asserts the **documented** divergence (`get_current_risk_free_rate(before-coverage)==0.05` while `MarketDataConnector().get_risk_free_rate(...)` is NaN); flip to assert `get_current_risk_free_rate(before-coverage)` is NaN (aligned).
- **Add (new engine-side behaviour tests — none exists today):**
  - A staleness test: a stub connector whose `get_iv_history` last row is `>30d` before `as_of`
    → `_resolve_pit_atm_iv` returns `None` (falls back), and `≤30d` → returns the IV.
  - A rate test: `as_of` before treasury coverage → the accessor returns NaN and the ranker
    treats it as no-rate.

### Ordering (critical)
Per the runbook #378 is **third** in the trio, **but** it must land **before Phase 0A** (the
frontier refresh that opens the IV↔spot gap) and certainly before Phase 2/3C consume the new
IV-term/skew data. If 0A is run first, co-refresh the served ATM IV in the same step.

---

## 4. §2 panel checklist (every (E) PR)

- [ ] Lane-claim block names the trio/gate file(s) touched (`docs/PARALLEL_SESSIONS.md` §5).
- [ ] §2 invariant intact: no path turns a non-tradeable candidate tradeable; no `EVEngine.evaluate` bypass.
- [ ] Downgrade-only contract intact (reviewers downgrade, never upgrade); dealer mult clamp `[0.70,1.05]` untouched.
- [ ] The matching characterization test is **flipped** (asserts the new behaviour), not deleted; new behaviour tests added where noted.
- [ ] `pytest tests/ -m "not backtest_regression"` green; `ruff check` + `ruff format --check` clean; coverage ≥ 80%.
- [ ] FILE_MANIFEST / TESTING.md updated for any new/renamed test; worklog fragment + INDEX regenerated.
- [ ] Held for review; operator verifies the panel before merge. **No autonomous land.**

## 5. After all three land → Phase R (single re-baseline)

Re-pin S27/S32/S34/S35 once (`--update-snapshot`), absorbing #372+#369+#378 `ev_mean` + the
Phase-0 data change; bump `EXPECTED_FRONTIER` + the data-test `FRONTIER` to 06-18; re-pick
W16/W30 if the frontier aged JPM out; flip the `#354` W2 PIT xfail only when **that** (Phase 3G)
lands; clear S34's provisional flag; operator-merge, no auto-update. (See
`docs/NEXT_DATA_SESSION_RUNBOOK.md` Phase 3/4 + `docs/WIRING_CAMPAIGN.md` Phase R.)

## 6. Cross-references

- `docs/WIRING_CAMPAIGN.md` — Phase 1 (this trio) + the #378-before-0A ordering + Phase R.
- `docs/NEXT_DATA_SESSION_RUNBOOK.md` — Phase 2 (2A #372 / 2B #369 / 2C #378) + the re-baseline rule.
- `docs/DATA_TEST_AUDIT_2026-06-09.md` — W17/W27/W36/W37 + C2 (R9 map) / C4.
- `tests/test_broad_pull_wiring_xfail.py` — the Phase 0B `xfail(strict)` acceptance scaffolds (#372 flips here).
- `CLAUDE.md` §2 + the `EnginePhaseReviewer` R1–R11 rules.
