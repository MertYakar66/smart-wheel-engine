# Data-layer Test Audit — 2026-06-09 (Phase 1 · discovery)

_origin/main @ `d0cdcde` · provider `MarketDataConnector` · frontier `2026-06-04` ·
read-only · decision trio untouched._

> A deeper, **test-coverage-focused** round building on the 2026-06-07 data+engine
> audit (`docs/DATA_ENGINE_AUDIT_2026-06-07.md`, weaknesses **W1–W13**) and the
> Phase-2 suites it spawned (`#358`, `#366`). It asks two questions the prior round
> did not answer systematically: **(a) which data→engine-output paths have a real
> test, and which are silent**, and **(b) what deeper data defects does the byte
> evidence show that no test yet pins.** Asserts nothing (Phase 2 turns the (T)
> items into tests). Every numeric claim is reproducible via
> `scripts/audit_data_engine.py` (W1–W13 re-run) and the new
> `scripts/audit_data_tests.py` (W14+ evidence), both run on `d0cdcde`.

---

## STATUS — EXECUTED (updated 2026-06-09): all (T) items landed, register extended W29–W37

This doc opened as Phase-1 discovery (register **W14–W28**, §5 below). It has since been
**executed and extended** in an autonomous session — all test-only **(T)** items
**W14–W37** are merged to `main` across **8 held, CI-green, self-reviewed PRs**; a
second discovery round (recon workflow) added **W29–W37**; and the engine/data-side
items are tracked as issues (not grabbed — they need the §2 lane-claim ceremony or a
producer change). Decision trio + data files untouched throughout; §2 only asserted,
never weakened.

**Landed test PRs (all squash-merged):**

| PR | Items | Surface |
|---|---|---|
| #370 | W14 served-IV gate · W18 PIT-IV vs real connector · W21 realized-vol positivity · W26 band constants · W27(T) fallback IV | IV |
| #371 | W15 EV sign controls · W16 real earnings lockout | data→engine |
| #373 | W19 yield-band + GICS-11 · W20 dividend_yield→carry · W17 R9 map characterization | fundamentals/sector |
| #374 | W22 depth invariant · W23 NaN-price two-sided · W25 dividend epsilon · W10 rate_1m | OHLCV/dividends |
| #375 | W24 credit ladder + Altman-Z | credit (off-EV) |
| #376 | W29 CC banded/finite · W30 CC real earnings lockout · W31 CC ex-div penalty sign · W32 CC EV sign | covered-call ranker |
| #377 | W33 full-universe bands at scale · W35 vix→R11 content · W34 liquidity | realism / peripheral |
| #379 | W36 vol_iv↔ohlcv date consistency · W37 rate-accessor divergence | cross-file / rate |

**Round-2 register (W29–W37)** — gaps the first round did not reach, found by the
round-2 recon workflow (covered-call ranker was real-data-starved vs the put side;
realism-at-scale; vix/liquidity content; cross-file/rate robustness). All landed (T)
above.

**Capability-map correction C3 (refines §1).** Round 1 said "VIX is off the EV verdict"
— true for the ranker's `ev_dollars` (the regime multiplier is the per-ticker OHLCV HMM,
not VIX). But VIX **is** EV-decision-relevant via **R11**: `candidate_dossier.py:546-553`
reads `dossier.vix_level` (← `get_vix_regime`, `wheel_runner.py:3490`) and downgrades
proceed→review when `vix_level > 25.0 & prob_profit > threshold`. So a wrong-scale VIX
silently breaks an EV-authoritative downgrade reviewer — pinned by **W35**.

**Tracked, NOT grabbed** (need the §2 ceremony / a producer change):

- **(E)** #369 — #363 IV gate doesn't clean the fundamentals-fallback IV path (W27 residue).
- **(E)** #372 — R9 sector cap uses the hardcoded `DEFAULT_SECTOR_MAP` (132/511), ignoring
  the pulled `gics_sector_name` (W17). HIGH; re-baseline-coupled.
- **(E)** #378 — engine robustness under `deep_history`/staggered refresh: IV-staleness
  gate (`_resolve_pit_atm_iv`) + rate-fallback divergence (`data_integration`) (W36/W37).
- **(D)** W28 (`edge_vs_fair` ≡ 0 until a market-mid premium producer); #354 (PIT
  fundamentals), #355 (blue-chip backfill), #357 (dividend producer clamp) — each already
  behind a behaviour-pinning `xfail(strict)`.

**Realism check (engine output is reliable/realistic).** Ranked UNIVERSE_100 at a
historical (2024-06-03) and the frontier (2026-06-04) as_of: **0 non-finite, 0 band
violations** (iv∈(0,3), prob∈[0,1], premium>0, strike<spot) at both. The 2024 skew
(84/86 negative-EV at a low-vol date) is the documented conservative
"forward-dist-lags-regime" bias (prior heavy-verify campaigns), not a new defect.

**Re-baseline note:** the slow backtest snapshots (S27/S32/S34/S35) carry an
`ev_mean`-only drift from #363's IV gate (a serving-logic re-pricing, NOT a regression —
trades/cash/NAV byte-identical); batched into the pending universe re-baseline. W16/W30
(JPM earnings-window) and W14–W37 magnitudes are FRONTIER-tied and re-baseline-aware.

---

## 0. Headline — the picture is *better* than the precedent implied

1. **Phase 2 (A)+(B) already shipped.** `tests/test_data_integrity_bloomberg.py`
   (real-CSV integrity) and `tests/test_data_to_engine.py` (real-data → ranker via
   `WheelRunner`, no §2 bypass) already cover most of the 2026-06-07 Phase-2 plan.
   **Of W1–W13, zero are open *uncovered* test gaps** — each is CLOSED, tracked by a
   behaviour-pinning `xfail(strict)`, or a positive control with a test (§4).
2. **#363 (`c2a0dbf`) neutralised W1/W8 at the data layer.** The connector gate
   `_clean_vol_iv_inplace` NULLs served IV outside `(3.0, 10000]` on every `vol_iv`
   read. On the committed monolith this NULLs **exactly the 17 sub-3.0 rows**
   (served put min = **3.127**, raw min = 0.01); the >10000 sentinel removes **0**
   (it lives only in the uncommitted deep panels). Issues `#356`/`#360` are CLOSED.
   **But the gate is asserted only on synthetic frames — never on the bundled read
   (W14, the marquee gap).**
3. **Two precedent capability-map claims are wrong** (§1): there is **no R0a credit
   gate** (the credit CSV is off the EV path), and **R9's sector cap uses a hardcoded
   `DEFAULT_SECTOR_MAP`, not `gics_sector_name`**.
4. **The genuine new (T) gaps are narrow and content-specific**: the served-IV gate
   pin, a real-data EV **sign** control, a real earnings→lockout wire-in, fundamentals
   yield/GICS bounds, credit ladder/Altman-Z plausibility, IV realized-vol
   positivity, and a per-name depth invariant. **15 new weaknesses (W14–W28); 13 are
   (T) landable now, 1 (E) and 1 (D) are tracked, not grabbed.**

---

## 1. Capability map (confirmed wiring + corrections)

Confirmed by reading the connector accessors and their EV-path consumers (file:line
refs from the recon). ✅ = precedent claim verified; ⚠️ = **correction**.

| Dataset → engine output | Wired via (verified) | Status |
|---|---|---|
| **OHLCV** → forward-dist cascade (empirical NOS→overlapping→block-bootstrap→HAR-RV) + RV-widening + per-ticker HMM regime | `get_ohlcv` (rename open→high/high→close/close→open) → `wheel_runner:1330 best_available_forward_distribution` → `forward_distribution.py:342-386`; HMM mult from log-returns `wheel_runner:1364`; → `ev_engine:386` | ✅ |
| **IV** (`vol_iv`) → BSM premium/Greeks/EV | `get_iv_history`; #363 gate `data_connector:317-334 _clean_vol_iv_inplace` band `(3.0,10000]` on monolith `:227` + assembled `:295`; PIT `_resolve_pit_atm_iv`; → `ev_engine:371-379 sigma=trade.iv` | ✅ |
| **earnings** → event lockout = **FIRST gate** in `EVEngine.evaluate` | `get_next_earnings` → `ScheduledEvent` `wheel_runner:1184` → `ev_engine:309-332` short-circuit **before** tx-cost/BSM/distribution | ✅ |
| **treasury** → BSM risk-free rate | `get_risk_free_rate` (**unconditional** `/100`, NaN on missing) `data_connector:782` → `ev_engine:376 r=trade.risk_free_rate` | ✅ |
| **dividends** (ex-div) → **covered-call** early-assignment EV **only** | `get_next_dividend` → CC ranker `wheel_runner:2519` → `ev_engine:410-416` (fires only `option_type=='call'`). Short-put ranker **never** consults ex-div (verified) | ✅ |
| **fundamentals** `eqy_dvd_yld_12m` → BSM carry `q` | `get_fundamentals` → `/100` `wheel_runner:1129` → `ev_engine:378 q=trade.dividend_yield` | ✅ |
| **fundamentals** `gics_sector_name` → ~~R9 sector cap~~ | feeds `screen_universe` + display **only** | ⚠️ **C2** |
| **credit** `sp_rating`/`altman_z_score` → ~~R0a credit gate~~ | `get_credit_risk` → `analyze_ticker.credit_rating` → legacy `_compute_wheel_score` + trade memo **only** | ⚠️ **C1** |
| **vix** → ~~EV regime~~ | `get_vix_regime` → strangle-timing score + display **only** | ⚠️ **C3** |

**Corrections (load-bearing for severity):**

- **C1 — No `R0a` credit gate exists.** `sp500_credit_risk.csv` is a *dead read off the
  EV path*: `get_credit_risk` has exactly one consumer (`analyze_ticker`,
  `wheel_runner:509`), feeding the legacy heuristic + memo display, **never**
  `rank_candidates_by_ev` / `EVEngine.evaluate`. The ranker's `credit_mult` comes
  from **FRED HY-OAS** (`BAMLH0A0HYM2`, `fred_adapter.py:53`), not this CSV. (The
  "credit gate" in `wheel_tracker` is the unrelated min-net-credit defensive-roll
  gate.) ⇒ credit-data defects are **display/legacy-screen severity**, not EV.
- **C2 — R9 sector cap uses a hardcoded `DEFAULT_SECTOR_MAP`** (`risk_manager.py:1755`),
  not the Bloomberg `gics_sector_name`. A ticker absent from the map collapses to an
  `'Unknown'` bucket, silently weakening the cap. The ranker's own `sector` output
  column is also from the map (`wheel_runner:1777`). ⇒ W17.
- **C3 — VIX is off the short-put/CC EV verdict** (the regime mult is the per-ticker
  OHLCV HMM, not VIX). No EV-path test should assert VIX moves `ev_dollars`.
- **C4 — `edge_vs_fair` ≡ 0 on the default synthetic-BSM path.** Premium and BSM fair
  are computed from identical inputs (`wheel_runner:1299` / `ev_engine:371-380`), so
  the documented VRP signal is **not measured** on the Bloomberg path; the wheel
  thesis survives only via the empirical forward distribution being narrower than
  IV-implied. ⇒ W28 (D, track).

---

## 2. What already exists (do **not** re-propose)

`tests/test_data_integrity_bloomberg.py` — real-CSV integrity (skips without data):
schema/required-columns (9 files); OHLCV positive prices, non-neg volume,
column-rename invariant, `(date,ticker)` uniqueness + no-future, dates-sorted-per-name,
NaN-prices `xfail(strict)` (4 rows, #357); raw-IV unit/band + zero put==call skew +
keys-unique; dividends materially-non-negative + referential; treasury band + 1994
coverage; seam membership/continuity + BK→BNY re-ticker; **fingerprint pins every
connector file** (fast-CI guard).

`tests/test_data_to_engine.py` — real-data → ranker (no §2 bypass): clean-universe
output **finite + banded** (`math.isfinite` ev_dollars/ev_raw, IV decimal, prob
bands, strike<spot, Wilson-CI coherence, valid tier); cascade tier (AAPL=NOS);
no-silent-drops accounting; thin-name graceful degradation; blue-chip backfill
`xfail(strict)` ×11 (#355); determinism; **DIS real ex-div → CC `expected_dividend`
(R1 mechanism)**; **W2 PIT behaviour `xfail(strict)`** (#354, #366 — selects the 2023
row, defeats a no-op `as_of`); negative control; 5-ticker smoke; full-universe slow
(produced==480 / dropped==31).

Synthetic-only (relevant): `test_data_connector.py::TestVolIvCleaningGate` (#363 gate,
tmp_path), `test_deep_iv_sentinel.py` (deep-only, `SWE_DEEP_TEST_DATA`-gated → skips
in CI), `test_ranker_iv_pit.py` / `test_mark_to_market_iv.py` (PIT IV via stubs),
`test_credit_rating_population.py` (R0a read via stub).

---

## 3. Coverage map (capability path → existing tests → silent gap)

| Capability path | Coverage | Silent gap → W |
|---|---|---|
| #363 served-IV band gate `(3.0,10000]` on bundled read | **synthetic-only** | No test calls `_load('vol_iv')`/`get_iv_history` on the bundled file asserting the served band → **W14 (HIGH)** |
| OHLCV integrity (price/volume/rename/keys/no-future/sorted) | **real-data** | per-name ≥504 **depth** not an invariant (only 11-name xfail, off-frontier skip) → **W22**; NaN-price xfail one-directional → **W23** |
| raw-IV unit/band/zero-skew/keys | **real-data** | asserts the **raw file**, not the served gate (slack bounds: danger≤50, out-of-band≤20) → companion to **W14** |
| IV realized-vol term structure (`volatility_30d/60d/90d/260d`) | **none** | no positivity/finiteness/ordering pin on the real columns → **W21** |
| ranker PIT-IV (`as_of` selects the right IV) | **synthetic-only (stubs)** | never validated vs real connector + bundled `vol_iv` → **W18** |
| served IV → ranker decimal range | real-data (`0<iv<3`) | band far wider than the served gate; a slipped sub-3 IV `/100`'d still passes → does not substitute for **W14** |
| fundamentals `eqy_dvd_yld_12m` bounds / GICS-11 set | schema-presence only | no value-band or GICS-11 membership pin on the real file → **W19** |
| `dividend_yield` → BSM carry `q` | synthetic / discrete only | never exercised end-to-end with **real** fundamentals → **W20** |
| R9 sector grouping (`DEFAULT_SECTOR_MAP`) | indirect | map-vs-served-universe coverage unpinned; `'Unknown'` collapse → **W17** |
| credit `sp_rating` ladder / `altman_z_score` plausibility | synthetic-only | no real-file ladder/Altman pin; CreditWatch suffixes pass un-normalised → **W24** |
| dividends ex-div → CC (R1) | **real-data (DIS)** | epsilon-negatives tolerated not clamped; `get_next_dividend` unclamped → **W25** |
| earnings → event lockout → `evaluate` | **synthetic dates only** | real `sp500_earnings.csv` → `get_next_earnings` → gate never wired on a real near-earnings name → **W16** |
| data → `EVEngine.evaluate` → EVResult **finite** | **real-data** | (covered) |
| data → EVResult **correct sign** | **none** | no known +EV / −EV real control; a sign inversion passes every test → **W15 (HIGH)** |
| two IV-sanity rules (connector band vs `utils.data_validation`) | each synthetic, uncrosschecked | opposite verdicts on the same value, undocumented → **W26** |

---

## 4. W1–W13 reconciliation (current main `d0cdcde`)

| W | Disposition on main | Test |
|---|---|---|
| **W1** IV `/100` heuristic | **MOOTED on serve** by #363 (served min 3.127, 0 ≤3.0); #356/#360 CLOSED; raw + dead-defensive trio line remain | EXISTS (raw band + synthetic gate). Residual **W27 (E)** below |
| **W2** dateless fundamentals/credit PIT | **OPEN (#354)** — now **behaviour-pinning** xfail (#366) | EXISTS (`xfail(strict)`) — fix is (D) |
| **W3** fingerprint completeness | **CLOSED** (fast-CI completeness guard) | EXISTS |
| **W4** seam universe gaps | Pinned structurally (joiners/leavers) | EXISTS |
| **W5** BK→BNY re-ticker | Pinned (presence/continuity); alias map = data follow-up | EXISTS |
| **W6** 17 thin <504 names | **OPEN (#355)** — 11 per-name xfail + graceful-degradation pin | EXISTS (`xfail`) — fix is (D) |
| **W7** dividends truncation | Resolved on main; R1 mechanism pinned | EXISTS |
| **W8** IV band extremes | **sentinel CLOSED** (served max 769%, 0 >10000); `(500,10000]` band intentionally kept + bounded | EXISTS |
| **W9** zero put==call skew | Pinned (==1.0 over 1,031,368 rows) | EXISTS |
| **W10** treasury negatives / rate_1m gap | **OPEN (#357)** band+coverage pinned (negatives allowed by design) | EXISTS (partial) — rate_1m pre-2001 NaN pin still MISSING → W-residual in PR-5 |
| **W11** dividend epsilon-negatives | **OPEN (#357)** materially-neg==0 pinned | EXISTS; clamp is (D). See W25 |
| **W12** branch-local staleness | Not a defect on main; FRONTIER-pinned fixtures | EXISTS |
| **W13** no silent drops | Positive control (fast + slow 480/31) | EXISTS |

**Net:** no W1–W13 item warrants a duplicate W14. The only stranded additive from
the old set is the **rate_1m pre-2001 NaN-not-0** pin (folded into PR-5).

---

## 5. New weakness register (W14–W28)

Each: evidence on the bytes (reproducible) · severity · **(T)** test-only (land now) /
**(E)** trio/engine change (track, do not grab) / **(D)** producer/data change (track).

### [HIGH] W14 · #363 served-IV band gate is untested on the bundled read — (T)
- **Evidence:** `conn._load('vol_iv')` over the bundled file → served put min **3.127**,
  max 769.273, cells ≤3.0 = **0**, cells >10000 = **0**; raw min 0.01 with **17** rows
  in `(0,3.0]` (HOLX×8, EA×3, BG/CEG/COF/HSY/UPS @0.01). The gate works but is pinned
  only on tmp_path frames (`TestVolIvCleaningGate`) + the CI-skipped deep test; the one
  real-file IV test reads the **raw** CSV via `pd.read_csv`, bypassing the connector.
- **Proposed (T):** in `test_data_integrity_bloomberg.py`, build `MarketDataConnector`,
  call `_load('vol_iv')` (and `get_iv_history` on HOLX), assert every non-null served
  IV ∈ `(3.0, 10000]` and that the 17 raw sub-3.0 rows are NULLed. (extends W1+W8)

### [HIGH] W15 · No real-data EV correct-**sign** control — (T)
- **Evidence:** real e2e tests assert finite + banded but never sign. `test_audit_viii_real_data_smoke`
  asserts only non-empty + field presence + `0<iv<3` + premium>0.05; `test_data_to_engine`
  asserts `math.isfinite` + bands. Probe: at as_of 2026-06-04 the transform is sign-bearing
  (XOM +112.86, UNH −77.35) — but nothing pins it; a data→forward-dist→EV sign inversion passes.
- **Proposed (T):** two pinned controls through `rank_candidates_by_ev` /
  `rank_covered_calls_by_ev` at a fixed as_of — a known +EV CSP → `ev_dollars>0`; a known
  structurally −EV trade (deep-delta CC on a fat-tail name) → `ev_dollars<0`. (new)

### [MEDIUM] W16 · Real earnings → event-lockout → `evaluate` never wired on the real file — (T)
- **Evidence:** all lockout tests use synthetic hand-built dates. Real `sp500_earnings.csv`
  (49,379 rows) has near-frontier names (BAC/C/FAST/GS/JPM/WFC within 40d of 2026-06-04);
  no test confirms the real file's date format + `'JPM UN'` suffix actually populate the
  gate and drop the name at `evaluate`. §2-adjacent (the first gate).
- **Proposed (T):** real near-earnings name at a fixed as_of → assert dropped `gate=='event'`
  with the gate on, produced with it off. (new)

### [MEDIUM] W17 · R9 sector grouping (`DEFAULT_SECTOR_MAP`) coverage vs served universe unpinned — (T)
- **Evidence:** C2 — `check_sector_cap` → `SectorExposureManager` → `DEFAULT_SECTOR_MAP`
  (`risk_manager.py:1755`), not `gics_sector_name`; names absent → `'Unknown'`. No test
  quantifies map coverage of `get_universe()` or pins `sector`-column == map.
- **Proposed (T):** assert every served-universe ticker has a non-`'Unknown'` map entry
  (or pin the gap count); pin `ranker['sector'] == DEFAULT_SECTOR_MAP.get(...)`, documenting
  the divergence from `fundamentals.gics_sector_name`. (extends W2/W4 / C2)

### [MEDIUM] W18 · Ranker PIT-IV never validated vs the real connector + bundled `vol_iv` — (T)
- **Evidence:** `test_ranker_iv_pit` / `test_mark_to_market_iv` use `_PitIVConn`/`_FakeIVConn`
  stubs with hand-fed dicts; real-data tests assert only `0<iv<3`, never that `iv` equals the
  as-of PIT value. A connector-side PIT regression (wrong `iloc[-1]`) is invisible.
- **Proposed (T):** real ticker (AAPL) at a fixed as_of → assert `rank(...).iv ≈
  (get_iv_history(...).iloc[-1][put,call].mean())/100`. (new)

### [MEDIUM] W19 · fundamentals `eqy_dvd_yld_12m` band + GICS-11 set unpinned on the real file — (T)
- **Evidence:** schema test checks column *presence* only. Real: yield ∈ [0.057%, 10.77%],
  median 1.875%, **95 NaN**, 0 neg, 0 >30; `gics_sector_name` = **exactly the 11 canonical
  GICS sectors** (0 outside). A 12th/typo'd sector would silently mis-bucket `screen_universe`.
- **Proposed (T):** assert non-NaN yield ∈ `[0,30]` and `set(gics_sector_name) ⊆ GICS_11`. (new)

### [MEDIUM] W20 · `dividend_yield` → BSM carry `q` never exercised with **real** fundamentals — (T)
- **Evidence:** every real-data EV test uses synthetic `dividend_yield` or the discrete
  ex-div path; the continuous `eqy_dvd_yld_12m` carry (e.g. AAPL 0.3416%) never pinned to flow
  into BSM `q`.
- **Proposed (T):** high-yield vs zero-yield real name → assert carry/fair shifts in the
  expected direction (confirms percent→decimal + wiring). (new)

### [LOW] W21 · IV realized-vol columns never sanity-checked on the real file — (T)
- **Evidence:** `volatility_30d/60d/90d/260d` are present (schema) + used as fixtures, but no
  positivity/finiteness/ordering pin on the real columns; they feed F4 RV-widening. Every
  "term_structure" test is VIX or SVI (synthetic).
- **Proposed (T):** assert the four realized-vol columns are positive + finite where present
  (soft: median curve plausibly ordered). (new)

### [LOW] W22 · Per-name OHLCV ≥504 depth not pinned as an invariant — (T)
- **Evidence:** the only depth assertion is `test_blue_chip_history_is_complete` (per-name
  `xfail` on 11 names, skips off-frontier). A *new* blue-chip truncated <504 bars outside the
  11-name set is invisible. Real: 511 names, depth min 7 / median 2117, **17** names <504.
- **Proposed (T):** pin the count of ≥504-bar names (re-derive) and/or `{<504 names} ⊆ ALL_THIN`. (extends W6)

### [LOW] W23 · NaN-price xfail is one-directional — (T)
- **Evidence:** `test_ohlcv_no_nan_prices` is `xfail(strict)` asserting `nan_rows==0` (4 known:
  BIIB 2020-11-06/2023-06-09, TPL 2019-05-16/2019-07-09). A 5th vendor-glitch NaN keeps the
  assert failing → xfail stays green, no signal.
- **Proposed (T):** add a two-sided pin `nan_rows == 4` (+ the 4 specific keys) so the count
  *growing* fails loudly, while the xfail still flips on the fix. (extends W11/#357)

### [LOW] W24 · credit `sp_rating` ladder + `altman_z_score` plausibility unpinned (off EV path) — (T)
- **Evidence:** real `sp_rating` = 21 distinct incl. **CreditWatch suffixes** `'A *-'`,
  `'BBB+ *-'`, `'CCC+ *+'`, plus `'NR'`, 52 NaN — pass through un-normalised. `altman_z_score`
  ∈ [−5.43, **129.5**], 75 NaN, 3 neg. No real-file pin. LOW because credit is off the EV path (C1).
- **Proposed (T):** after stripping ` *-`/` *+`, assert `sp_rating ∈ S&P_ladder ∪ {NR}`; assert
  Altman ∈ wide band `[-10,200]` + pin the negative count (3). (new)

### [LOW] W25 · Dividend epsilon-negatives tolerated, not clamped; `get_next_dividend` unclamped — (T)
- **Evidence:** 82 epsilon-negatives (min −2.4e-14, all ∈ `[-1e-9,0)`); `test_dividends_nonnegative`
  tolerates `< -1e-9`. `get_next_dividend` has no clamp; only the positive DIS case is tested.
- **Proposed (T):** connector-level test (synthetic −2e-14 fixture) that `expected_dividend` never
  goes negative. Producer clamp = (D) #357. (extends W11/#357)

### [INFO] W26 · Two divergent IV-sanity rules coexist uncrosschecked — (T)
- **Evidence:** connector band `(3.0,10000]` (nulls ≤3% as garbage) vs `utils.data_validation`
  (`>10 → /100`, accepts 1–5 as "high but valid") — opposite verdicts on a 2% IV. Different data
  paths today (Bloomberg-served vs Theta-chain), but a latent trap.
- **Proposed (T):** document the connector gate as authoritative for Bloomberg-served IV; note
  `utils.data_validation` is Theta-chain-only (or add a consistency test if they can overlap). (new)

### [LOW] W27 · W1 heuristic still load-bearing for the fundamentals-fallback IV path — **(E) track**
- **Evidence:** `_clean_vol_iv_inplace` runs only for `key=='vol_iv'`; the ranker IV fallback
  (`wheel_runner:1082-1111`) reads `implied_vol_atm`/`volatility_30d` from `sp500_fundamentals.csv`
  (uncleaned) and applies the `if iv>3.0: iv/=100` heuristic. #363 does **not** clean this path.
- **Disposition:** track — cleaning fundamentals IV at the connector is a trio/engine change
  (needs the §2 lane-claim ceremony). A (T) test can *pin* that the fallback IV is percent and the
  heuristic is its only normaliser. (extends W1)

### [INFO] W28 · `edge_vs_fair` is structurally 0 on the synthetic-BSM path — **(D) track**
- **Evidence:** C4 — premium and BSM fair from identical inputs → `edge_vs_fair == 0` by
  construction; the VRP signal is dead until a real market-mid premium producer lands (the
  Bloomberg connector lacks an option-chain premium source).
- **Disposition:** track — a data/producer change. Coverage agents must not treat `edge_vs_fair`
  as a live VRP signal on the Bloomberg path. (new)

---

## 6. Ranked (T) plan — surface-grouped, one PR per surface

Each PR is additive, real-CSV (reuse the `HAS_BLOOMBERG_DATA` skipif), held for review,
CI-green, with a worklog + FILE_MANIFEST coverage. Synthetic-fixture assertions for
logic; the live CSVs only for "is the served data sane". Confirmed-broken paths →
`xfail(strict=True)` pinning **behaviour** (the #366 lesson), never a signature proxy.

| PR | Surface | Lands | Why first |
|---|---|---|---|
| **PR-1** | **IV surface** | W14 (served-gate on bundled read) + raw-vs-served clarification + W18 (PIT-IV vs real connector) + W21 (realized-vol positivity) + W26 (note) | The #363 gate is the marquee gap — a new connector behaviour with **zero** real-data coverage; highest blast radius (IV → every premium/EV) |
| **PR-2** | **data→engine e2e** | W15 (EV **sign** controls) + W16 (real earnings→lockout) | §2-adjacent: the sign of the authoritative transform + the first gate, on real bytes |
| **PR-3** | **fundamentals / sector** | W19 (yield band + GICS-11) + W20 (dividend_yield→carry real) + W17 (DEFAULT_SECTOR_MAP coverage) | Closes the carry-`q` and R9-grouping silent paths; encodes correction C2 |
| **PR-4** | **OHLCV / dividends hygiene** | W22 (depth invariant) + W23 (NaN-price two-sided) + W25 (epsilon clamp) + W10 rate_1m NaN residual | Cheap, hardens the integrity suite's one-directional/slack pins |
| **PR-5** | **credit (off-EV-path)** | W24 (rating ladder suffix-strip + Altman band) | Lowest stakes (display/legacy only, C1); land last or fold into PR-4 |

**Tracked, NOT grabbed** (need the §2 lane-claim or a producer change): W27 (E,
fundamentals-fallback IV clean), W28 (D, market-mid premium), and the existing
data-layer fixes behind their behaviour-pinning xfails — #354 (W2), #355 (W6),
#357 (W10/W11 producer clamp).

---

## 7. Reproduce

```
# W1–W13 byte-state on main (provider/frontier logged, no §2 bypass):
py -3.12 scripts/audit_data_engine.py --universe audit --json out.json

# W14+ deeper evidence (served-IV gate, GICS/yield, rating ladder, Altman-Z,
# per-name depth, NaN rows, e2e finite+sign):
py -3.12 scripts/audit_data_tests.py --json out2.json
```

Both run read-only against `data/bloomberg/` on `origin/main @ d0cdcde`; connector
logged as `MarketDataConnector`; every engine probe routes through
`WheelRunner.rank_candidates_by_ev` (no §2 bypass). The decision trio
(`ev_engine.py` / `wheel_runner.py` / `candidate_dossier.py`) was read only.

---

_Phase 1 ends here. Hold for review before Phase 2._
