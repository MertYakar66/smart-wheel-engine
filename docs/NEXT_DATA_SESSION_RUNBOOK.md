# Re-baseline session — single authoritative execution runbook

> **The one checklist for the supervised Bloomberg re-baseline session.**
> Consolidates the open data queue (#339, #355, #354, #357) **and** the three
> (E) trio/risk-gate fixes (#372, #369, #378) into one ordered, end-to-end plan.
> The governing rule: **everything that moves the frontier or the EV output must
> land BEFORE the 4-snapshot re-pin** — so the single re-baseline captures the
> data change, the #363 IV-gate re-pricing, *and* the (E) fixes' EV-output impact
> in one pass (don't pay the ~4 h re-baseline tax twice).
>
> Updated 2026-06-09 against `main @ 9847edc` (after the 2026-06-09 data-test
> audit landed W14–W37 + issues #369/#372/#378 — see
> `docs/DATA_TEST_AUDIT_2026-06-09.md`). Original turnkey data-queue runbook
> generated 2026-06-08 against `main @ 67b57fc` (#365). Re-run the seam/thin-history
> audit after any future refresh — newly-added constituents tend to land
> recent-only and add rows to these tables.

**Prerequisite:** a **logged-in Bloomberg Terminal** with `xbbg` reachable for
Phase 1A and Phase 5 (the Cowork sandbox carries only the committed CSVs; see
`docs/DATA_POLICY.md` §7). Phase 1B (git integration), **Phase 2 (the three (E)
fixes)**, and Phase 3/4 (re-baseline + verify) need **no Bloomberg** and can be
done from a clean clone — but Phase 2 must land **before** Phase 3.

**Standing constraints — do not relax (CLAUDE.md §2/§3):**
- The three (E) fixes are **decision-layer / risk-gate** changes. Each gets the
  full **§2 lane-claim ceremony**, is **held for review** (the operator verifies
  the §2 panel as each lands), and preserves the **downgrade-only reviewer
  contract** + the **`EVEngine.evaluate` invariant**. **No (E) fix lands
  autonomously.**
- Re-baseline is **operator-merge, no auto-update** of snapshots.
- Branch + PR for every change; trio PRs lane-claimed; never commit to `main`.

---

## Execution order at a glance (strict)

The ordering *is* the plan. Run top-to-bottom; **do not re-pin the snapshots
(Phase 3) until everything above the re-pin has landed**, or you re-baseline
twice.

| # | Phase | What | Bloomberg? | §2 ceremony? |
|---|---|---|---|---|
| **1** | **Universe data** | A: CASY + 10 blue-chip backfills · B: BK↔BNY collapse, dividends union, `UNIVERSE_100` re-derive | A: **yes** / B: no | data/universe — no §2 bypass |
| **2** | **(E) trio/risk-gate fixes** | **#372** (HIGH, R9→GICS) → **#369** (IV-gate fallback clean) → **#378** (IV-staleness gate + rate-fallback) — each lane-claimed + held for review | no | **yes — full ceremony each** |
| **3** | **Re-baseline + verify** | re-pin S27/S32/S34/S35 (captures #363 `ev_mean` + the (E) frontier impact in **one** pass), bump `EXPECTED_FRONTIER`, flip per-name xfails, clear S34's provisional flag | no | §2 panel on the universe shift |
| **4** | **Frontier-coupled test re-picks** | re-pick W16/W30's JPM earnings-window names (**only if the frontier moved**); re-pin the full-universe 480/31 split | no | — |
| **5** | **(D) producer pulls** | fold **#354 / #355 / #357 / W28** into the **same** Terminal session (don't make a second trip) | **yes** | #354 also unlocks a separate trio PR |

> Phase 5's #355 (blue-chip backfills) is already Phase 1A — listed again only so
> the operator pulls *all* producer-gated items in the one Terminal session.

---

## 0. The one distinction that drives the data half

Most of the **data** queue is **NOT** Bloomberg-gated. Only the items in
**Phase 1A** (and Phase 5's residual pulls) need the Terminal; everything in
**Phase 1B** is reconstructable from data already in git (per
`docs/CASY_BACKFILL_SPEC.md`, verified on the bytes 2026-06-07). The **(E) fixes
(Phase 2)** are not data at all — they are trio/risk-gate code, gated only by the
§2 review, but they change EV output so they must precede the re-pin.

| Truly Bloomberg-gated (Phase 1A / 5) | Git-reconstructable / local (Phase 1B) | Code, §2-gated (Phase 2) |
|---|---|---|
| CASY pre-2026 `ohlcv`/`vol_iv`/`liquidity`/`earnings` (#339, #355) | BK↔BNY entity collapse (#339) | #372 R9 sector cap → real GICS |
| 10 other blue-chip OHLCV backfills (#355) | `sp500_dividends.csv` union (`CTRA/LW/MTCH/PAYC`, #339) **+ epsilon-clamp** (W25/#357) | #369 #363 IV-gate fundamentals-fallback clean |
| Refresh of the 3 script-producible files to the new frontier (optional) | Fragment integration + de-dup into the monoliths | #378 IV-staleness gate + rate-fallback divergence |
| #354 dated (PIT) fundamentals panel — larger data-model change | `UNIVERSE_100` re-derivation (#339) | |
| #357 W10 `rate_1m` pre-2001 coverage — optional, low-pri | | |

> If a Terminal is genuinely unavailable, **Phase 1B + Phase 2 are still real
> forward progress** (BK↔BNY collapse + dividends union are pure git; the three
> (E) fixes need no Bloomberg) — **but the snapshot re-pin (Phase 3) must wait for
> Phase 1A**, or it runs twice. Do Phase 2 ahead of the Terminal session only if
> you will then hold the re-baseline until the data lands.

---

## Phase 1 — Universe data

### Phase 1A — Bloomberg pulls (Terminal required)

Pull **fragment CSVs (the affected tickers' rows only)** — do **not** edit the
big monoliths on the pull box. Push/send the fragments; integration + de-dup is
Phase 1B, done here on a reviewed branch. Match each connector file's exact
schema and ticker format.

**A1. CASY — the authoritative spec already exists.**
`docs/CASY_BACKFILL_SPEC.md` is the canonical, on-the-bytes spec for CASY's
**4 files** (`ohlcv`, `vol_iv`, `liquidity`, `earnings`), with exact `xbbg`
snippets, the `CASY UW Equity` ticker, the `2018-01-02 → 2026-06-04` range, the
schema-per-file gotchas (e.g. liquidity's 3rd column is `shares_out` =
`EQY_SH_OUT`, not `bid_ask_spread`), and its validation gates (overlap to the
cent on the 52 existing `2026-03-23→` rows; verify the IV field against an
existing name first). **Follow that doc for A1 verbatim.**

**A2. The other 10 blue-chip OHLCV backfills (#355).**
Same fragment-pull pattern as CASY's `ohlcv` step (`scripts/pull_ohlcv.py`'s
method: split-adjusted `blp.bdh` of `PX_OPEN/HIGH/LOW/LAST/VOLUME`, long format,
schema `date,ticker,open,high,low,close,volume`, ticker `"<TICKER> <exch> Equity"`).
Range `2018-01-02 → 2026-06-04` (overlap the existing tail rows for validation).
These carry `< 504` OHLCV bars on `main` so the ranker's `min_history_days=504`
gate drops them, though their real history is far longer:

| ticker | first bar (main) | bars | real history |
|---|---|---|---|
| WMT | 2025-12-09 | 122 | public since 1972 |
| KMB | 2025-05-30 | 255 | decades |
| CPB | 2024-08-19 | 450 | decades |
| DPZ | 2025-01-02 | 356 | IPO 2004 |
| PLTR | 2024-11-26 | 380 | IPO 2020 |
| VEEV | 2026-03-23 | 52 | IPO 2013 |
| COHR | 2026-03-23 | 52 | since 1987 (II-VI) |
| LITE | 2026-03-23 | 52 | IPO 2015 |
| SATS | 2026-03-23 | 52 | since 2008 |
| VRT | 2026-03-23 | 52 | since 2020 |

(CASY is the 11th name in #355's table — covered by A1.) **Also check `vol_iv`
coverage** for any name that is *also* IV-thin and backfill it the same way
(CASY's `vol_iv` step in the spec is the template); the OHLCV gate is the
headline, but a name needs IV to be fully tradeable.
**Do NOT** backfill the genuinely-recent names (BNY/FDXF/SNDK/SW/PSKY/Q) — those
are real new constituents, pinned by graceful-degradation tests, not defects.

**A3 (optional). Refresh the 3 script-producible files to the new frontier.**
Only `sp500_ohlcv.csv`, `sp500_liquidity.csv` (xbbg → Terminal), and
`treasury_yields.csv` (yfinance, no Terminal) have in-repo producers. If you
also want the frontier moved past `2026-06-04`:
- `scripts/pull_ohlcv.py` — **edit the hardcoded `end_date="2026-03-20"` at line 19** first.
- `scripts/pull_liquidity.py` — **edit `end_date="2026-03-20"` at line 26** first.
- `python scripts/pull_treasury_yields_yf.py --incremental` (already current; no Terminal).
The other 6 connector files (incl. the core `sp500_vol_iv_full.csv`) have **no
in-repo producer** — see `docs/bloomberg_refresh_runbook.md`. **If the frontier
moves, it cascades: Phase 3 bumps `EXPECTED_FRONTIER` + the data-test `FRONTIER`
constants, and Phase 4 re-picks W16/W30.**

### Phase 1B — git-reconstructable integration (no Bloomberg, reviewed branch)

> **Phase 1A is DONE (2026-06-17 supervised session).** The CASY + 10 blue-chip
> fragments **and** the #354 PIT panel are pulled, validated, and pushed —
> staged-only — on branch **`claude/phase1a-casy-bloomberg-pull` @ `bbc9c91`**
> (`staging/casy/`, `staging/blue_chips/`, `staging/fundamentals_pit/`).
> **Fetch that branch** to get the fragments; manifest + validation/vintage +
> per-file notes in `docs/worklog/phase1a-bloomberg-fragments-handoff.md` and the
> `staging/*/PULL_NOTES.md`. The raw fragments are intentionally **not** PR'd to
> `main` — *this* integration is their merge vehicle.

Per `docs/CASY_BACKFILL_SPEC.md` §"After you push/send the fragments":

1. **Integrate fragments** — fold the CASY + 10 blue-chip fragments into the
   four monoliths (`sp500_ohlcv.csv`, `sp500_vol_iv_full.csv`,
   `sp500_liquidity.csv`, `sp500_earnings.csv`), **de-duping** the existing
   recent rows against the overlap window.
2. **BK↔BNY collapse** — fold `BNY` into `BK`'s continuous history so the
   connector sees **one** entity (BK OHLCV runs full `2018→2026-03-20`; `BNY`
   re-tickers the rest; `BNY UN` dividends already carry BK's full history). This
   frees the phantom universe slot. *(Pure git — no Bloomberg.)*
3. **Dividends file — union + epsilon-clamp (both pure-git byte rewrites of the
   same file; do them together, *before* the re-pin).**
   - **Union** — `sp500_dividends.csv` = refresh ∪ main's
     `CTRA UN / LW UN / MTCH UW / PAYC UN` rows (the refresh dropped their
     2018–2024 history; `MTCH` was the one #339 originally missed). Current
     **and** complete.
   - **Epsilon-clamp (W25 / #357 — moved here from Phase 5).** Clamp the 82
     epsilon-negative `dividend_amount` rows (~−2.4e-14 float noise on
     Discontinued/Omitted rows) to 0. It rewrites `sp500_dividends.csv` bytes, so
     by the **#340 raw-byte fingerprint rule** (Phase 3 step 5) it **must** land
     here with the union — deferring it to Phase 5 would re-trip the fingerprint
     guard *after* the Phase-3 re-baseline and force a second ~4 h re-pin. The
     W11/W25 `xfail` flips green on this clamp. *(The other half of #357 —
     `rate_1m` pre-2001 coverage — is genuinely producer-gated + pre-window, so it
     correctly stays in Phase 5.)*
   *(Both pure git — no Bloomberg.)*
4. **Re-derive `UNIVERSE_100`** — after B2/B3, regenerate the constant in
   `backtests/regression/universes.py` from
   `MarketDataConnector().get_universe()[:100]` (expect `CMG`/`CMI` to **return**
   as `BNY` collapses into `BK`, and CASY to become a legitimate in-window
   member). `test_universes_match_connector` enforces the derivation matches the
   connector — run it green before proceeding.

---

## Phase 2 — the three (E) trio/risk-gate fixes (§2 ceremony, held for review)

These are **decision-layer / risk-gate** code changes, not data. They change EV
output, so they land **after** the universe data (Phase 1) and **before** the
re-pin (Phase 3) — the re-baseline then absorbs their impact in the same pass.
Order is **#372 → #369 → #378**. Each: separate lane-claimed trio PR, **held for
review**, with a §2 panel the operator verifies before merge. None weakens the
downgrade-only contract or bypasses `EVEngine.evaluate`.

### 2A. #372 (HIGH) — R9 sector cap → real `gics_sector_name`
- **Finding (W17 / capability-correction C2, audit §1):** `check_sector_cap` →
  `SectorExposureManager` binds `self.sector_map = sector_map or DEFAULT_SECTOR_MAP`
  (`engine/risk_manager.py:1755`; the literal at `:1579`), **not** the pulled
  `gics_sector_name`. **132/511** served names are in the map → **379 collapse to
  the `'Unknown'` bucket**, silently weakening the R9 cap. The ranker's own
  `sector` output column is also from the map (`wheel_runner:1777`).
- **Fix direction:** read `fundamentals.gics_sector_name` as the primary sector
  source for the R9 gate; **use W19's GICS-quality finding for the fallback** —
  the pulled column is **exactly the 11 canonical GICS sectors** with **95 NaN**
  (recent adds / seam-leavers / names without a fundamentals row). So: GICS
  primary → `DEFAULT_SECTOR_MAP` for the 95 NaN / seam-leavers → only then a
  **counted** `'Unknown'` (never a silent collapse). Pin the canonical-11 set
  (W19's `GICS_11`).
- **Test impact — W17 flips.** `test_data_to_engine.py::test_r9_sector_map_ignores_pulled_gics_characterization`
  *characterizes* today's DEFAULT_SECTOR_MAP behaviour; when #372 lands it must
  flip to assert the **GICS-primary** path (and the counted-`'Unknown'` fallback).
  Update it in the #372 PR, not separately.
- **Depends on:** W19 (GICS-11 quality / NaN coverage), already landed (#373).
- **§2:** R9 is a downgrade-only soft-warn cap (CLAUDE.md §2) — the fix must keep
  it downgrade-only and keep the same `enforce_*_cap` semantics; full §2 panel.

### 2B. #369 — #363 IV gate does not clean the fundamentals-fallback IV path
- **Finding (W27, audit §5):** `_clean_vol_iv_inplace` runs only for
  `key=='vol_iv'`. The ranker's IV **fallback** reads `implied_vol_atm` /
  `volatility_30d` from `sp500_fundamentals.csv` **uncleaned**
  (`wheel_runner.py:1082`, with the inline `if iv > 3.0: iv/=100` heuristic at
  `:1101`; mirrored on the CC path at `:2418`). So the #363 connector gate is
  **not** the safety net for the fallback path — only the inline heuristic is.
- **Fix direction:** clean the fundamentals-served IV at the connector (extend
  the #363 gate to the fundamentals accessor), so a sub-3.0 / sentinel value is
  NULLed before it reaches the ranker, and the inline heuristic stops being the
  sole normaliser.
- **Test impact:** W27's two characterization tests
  (`test_fundamentals_fallback_iv_input_is_percent`,
  `test_363_gate_does_not_clean_fundamentals_iv`) pin **today's** uncleaned
  behaviour; when #369 lands they flip to assert the gate now cleans the
  fundamentals path. Update them in the #369 PR.
- **§2:** connector-layer serving change feeding `EVEngine.evaluate`; no verdict
  logic — but it moves EV magnitudes (hence ordered before the re-pin). §2 panel.

### 2C. #378 — IV-staleness gate + rate-fallback divergence
- **Finding (W36 / W37, audit §5, round 2):**
  - **W36:** `_resolve_pit_atm_iv` (`wheel_runner.py:153-202`) takes
    `get_iv_history(end_date=as_of).iloc[-1]` with **no staleness gate** — the
    spot path *has* a 30-day gate. A refresh shipping IV staler than OHLCV for a
    name would silently price BSM against stale IV. Latent today (max IV↔OHLCV gap
    = **1 day** across 509 common names) → bites under a **staggered / deep_history
    refresh**, which this session may trigger.
  - **W37:** the EV-path rate accessor
    `engine.data_integration.get_current_risk_free_rate` (wired at
    `wheel_runner.py:588-590`, feeding BSM) returns a **silent `0.05`**
    (`data_integration.py:323`+) for an `as_of` **before** treasury coverage,
    diverging from the connector's NaN-on-missing contract.
- **Fix direction:** add a staleness gate to `_resolve_pit_atm_iv` mirroring the
  spot path's 30-day rule; make the rate fallback consistent (NaN-or-explicit,
  not a silent 0.05) or document the divergence as intentional with a guard.
- **Test impact:** W36/W37 are data-side cross-file pins today; the engine-side
  gates are this fix. Add behaviour tests for the new gate (do not loosen W36/W37).
- **§2:** `_resolve_pit_atm_iv` feeds the authoritative BSM/EV path; §2 panel.

---

## Phase 3 — re-baseline + verify

> Now captures **three** things in one pass — the Phase-1 universe change, the
> **#363 IV-gate `ev_mean` re-pricing** already latent on `main` (a serving-logic
> re-pricing, NOT a regression: trades/cash/NAV byte-identical — see
> `backtest-regression-slow-lane-drift` memory + audit "Re-baseline note"), and
> the **Phase-2 (E) fixes' EV-output impact**. This is *why* Phases 1–2 precede
> the re-pin.

5. **Expect the drift guard to go red first (this is the signal, not a bug).**
   `tests/test_backtest_regression.py::test_snapshot_data_fingerprint_matches_current`
   (the #340 guard) compares each snapshot's pinned `connector_data_sha256`
   against the live connector set and **fails the moment the data changes** —
   forcing an explicit re-baseline before the multi-hour markers run. (Note: the
   fingerprint hashes **raw bytes**, so it catches the Phase-1 data change but
   **not** the #363 serving-logic re-pricing — that one surfaces only as the
   `ev_mean`-only marker drift on S27/S32/S34, caught at unit speed by the W14
   served-output test. Both are absorbed by the single re-pin here.) Diagnose
   first per `TESTING.md` §"Backtest regression — re-baseline workflow".
6. **Re-baseline all four snapshots** (~4 h total). A dividends change moves the
   covered-call realized cash on *all four* (not just S34) — proven in R1 — and
   the (E) fixes + #363 re-pricing move `ev_mean` across the board, so regenerate
   every one:
   ```
   python -m backtests.regression.s27_ivpit_24t_100k   --update-snapshot
   python -m backtests.regression.s32_friction_24t_1m  --update-snapshot
   python -m backtests.regression.s34_universe_100t_1m  --update-snapshot
   python -m backtests.regression.s35_oos_24t_100k     --update-snapshot
   ```
   Capture via a **full-metric diff harness** (not the first-failing assert);
   amend each snapshot's note preserving the original numbers + a per-snapshot
   attribution line (universe Δ / #363 `ev_mean` / which (E) fix); file a
   snapshot-update record (see `TESTING.md` step 4). Tag **RA-style / worklog,
   NOT a D-number** (D-series live at D23).
7. **Confirm the markers** against the new baselines (~2.5 h):
   ```
   pytest tests/test_backtest_regression.py -m backtest_regression
   ```
   All four must pass (byte-identical to the regenerated snapshots). (Gate other
   pre-push runs with `pytest tests/ -m "not backtest_regression"` — the bare
   `pytest tests/` **hangs hours** on S34; see `TESTING.md` / #367.)
8. **Bump `EXPECTED_FRONTIER`** in `tests/test_preflight_environment.py` to the
   new OHLCV frontier — **only if A3 moved it** — in the same commit as the
   refresh (the guard-rot rule from `docs/DATA_POLICY.md` §5). **Also bump the
   data-test `FRONTIER` constant** (`tests/test_data_to_engine.py:39` **and**
   `tests/test_data_integrity_bloomberg.py`) — the W14–W37 suite is FRONTIER-pinned
   and will otherwise skip or assert stale dates. CASY/blue-chip backfills extend
   *history*, not the frontier, so they alone do **not** require a bump.
9. **Flip the xfail trackers** as each lands (`xfail(strict=True)` → green):
   - `tests/test_data_to_engine.py::test_blue_chip_history_is_complete[<ticker>]`
     — one per backfilled name (#355).
   - `tests/test_data_to_engine.py::test_fundamentals_credit_are_point_in_time`
     — only when the #354 PIT accessor lands (separate trio PR; Phase 5 unlock).
10. **Clear S34's ⚠️ provisional flag** (set in PR #338) and run the **§2 review
    panel** on the connector/universe change *and on each Phase-2 (E) fix* — the
    universe shift changes the backtest universe but `EVEngine.evaluate` is
    untouched by the data change; the (E) fixes touch the trio/risk-gate and each
    carry their own panel (no §2 bypass; downgrade-only preserved).

---

## Phase 4 — frontier-coupled test re-picks

11. **Re-pin the full-universe split.** `test_data_to_engine.py::test_full_universe_no_silent_drops_and_split`
    pins `produced == 480` / `dropped == 31` at the 2026-06-04 frontier. The
    Phase-1 universe change re-pins this **even without a frontier move** (CASY
    becomes an in-window member, `CMG`/`CMI` return, BK↔BNY collapses one slot).
    Re-derive and update both counts; the pin going red is the signal, by design.
12. **Re-pick W16/W30's JPM earnings-window names — only if the frontier moved.**
    `test_real_earnings_event_lockout_fires` (W16, put, 35-DTE) and
    `test_cc_real_earnings_event_lockout_fires` (W30, CC, 49/63-DTE) pin **JPM**
    purely because its earnings fall inside that window at the *current* frontier
    (2026-06-04). If the new frontier ages JPM out, both go RED **benignly** — it
    is NOT a broken event-gate wire. Fix = re-pick a near-earnings name (probe
    `get_next_earnings` within ~40 d of the new frontier; at 2026-06-04 the set
    was **BAC / C / FAST / GS / JPM / WFC**), not a code change. **W15/W32 sign
    controls (XOM/HD>0, UNH/AAPL<0) are sign-only → robust to the frontier move;
    no re-pick.** (See `backtest-regression-slow-lane-drift` memory.)

---

## Phase 5 — fold the (D) producer pulls into the SAME Terminal session

Do these in the **one** logged-in session (don't make a second trip). All are
producer/data changes tracked behind behaviour-pinning `xfail(strict)`:

- **#355** — the 11 blue-chip backfills (already executed in Phase 1A; listed
  here only so the operator confirms all producer-gated pulls are done together).
- **#354 — PIT fundamentals. DATA PULLED (2026-06-17) — accessor still pending.**
  The dated panel is done: `staging/fundamentals_pit/sp500_fundamentals_pit.csv`
  (monthly, 2015-01-02→2026-05-31, 503 names, 72,937 rows) carries the EV-consumed
  fields incl. `eqy_dvd_yld_12m` → BSM carry `q`, on branch
  `claude/phase1a-casy-bloomberg-pull @ bbc9c91`. It closes the lookahead in the
  single dateless `sp500_fundamentals.csv`. **Still to do (no Bloomberg):** the
  **separate trio PR** adding `as_of` to `get_fundamentals`/`get_credit_risk`,
  threaded from `wheel_runner` (decision-layer, §2 review); the W2 PIT xfail
  (`test_fundamentals_credit_are_point_in_time`, #366) flips only when that
  accessor lands. **Before that accessor lands, confirm the panel's ~31%
  `eqy_dvd_yld_12m` NaN is "legitimately no-dividend / not-yet-listed" vs "missing
  data"** (coverage is 69.2%; that field feeds carry `q`, so silent-missing would
  distort EV — expected to be non-payers + early-window unlisted names, but verify).
- **#357 — W10 `rate_1m` pre-2001 coverage → WON'T-FIX (closed 2026-06-17, not a gap).**
  `treasury_yields.csv` `rate_1m` is non-null from exactly **2001-07-31**, which is
  the genuine inception of the US Treasury 1-month constant-maturity series — it did
  **not exist** before then. So the pre-2001 NaN (23.4%) is **correct**, not missing
  data; "filling" it would mean splicing in a *different* proxy series (e.g. 4-week
  bill), it is entirely pre-backtest-window, and it has no engine consumer (the
  integrity band-pin already passes). **Do not re-attempt this pull.** *(This closes
  only the `rate_1m` half of #357. The other half — the dividend epsilon-clamp
  (W11/W25) — is still live and **moved to Phase 1B step 3**: a pure-git byte rewrite
  of `sp500_dividends.csv` that must land before the re-pin, else it re-trips the
  #340 fingerprint.)*
- **W28 — `edge_vs_fair` ≡ 0 (D, BLOCKED here).** This needs a **market-mid
  option-premium producer** that the Bloomberg connector **does not have** (C4 —
  premium and BSM fair are computed from identical inputs, so the VRP signal is
  dead by construction). It **cannot be cleared by a Bloomberg pull** — left
  tracked; flagged so coverage agents never treat `edge_vs_fair` as a live VRP
  signal on the Bloomberg path.

---

## Validation gates

**Data (from `docs/CASY_BACKFILL_SPEC.md`):**
- **CASY OHLCV overlap** (`2026-03-23 → 2026-06-04`) matches the 52 existing
  committed rows **to the cent**.
- **vol_iv implied-vol field** verified against an existing name (e.g. `AAPL UW`)
  over a recent month before trusting any backfilled IV (Bloomberg has **no
  put/call skew** — `hist_put_imp_vol == hist_call_imp_vol` exactly; see memory
  `bloomberg-iv-no-skew`).
- **Seam audit** post-integration shows **zero** recent-only (0-in-window) names
  left in `UNIVERSE_100`.
- `test_universes_match_connector` **green**.
- **W14 served-IV band** still green after any IV change (the cheap guard for the
  #363-class serving re-pricing).

**(E) fixes (Phase 2) — each PR:**
- Full **§2 review panel** (operator-verified); downgrade-only contract intact;
  no `EVEngine.evaluate` bypass.
- The matching characterization test **flipped** (W17 for #372, W27 for #369),
  not deleted; new behaviour tests added for #378's gates.
- `pytest tests/ -m "not backtest_regression"` green; ruff clean.

**Re-baseline (Phase 3/4):**
- All four markers byte-identical to the regenerated snapshots.
- Full-universe 480/31 split re-pinned; W16/W30 green (re-picked iff frontier moved).
- `EXPECTED_FRONTIER` + data-test `FRONTIER` consistent with the served frontier.

---

## Cross-references

- `docs/DATA_TEST_AUDIT_2026-06-09.md` — the W14–W37 register + the (E)/(D)
  dispositions (§5) + capability corrections C1–C4; reproducible probe
  `scripts/audit_data_tests.py`.
- `docs/CASY_BACKFILL_SPEC.md` — authoritative CASY pull spec (Phase 1A) + the
  reconstructable-vs-Bloomberg breakdown.
- `docs/bloomberg_refresh_runbook.md` — per-file producer reality (which of the 9
  connector files have a script vs. need recovered BQL).
- `docs/DATA_POLICY.md` §5 — refresh procedures + the `EXPECTED_FRONTIER`
  bump-on-refresh rule.
- `TESTING.md` §"Backtest regression — re-baseline workflow" — the canonical
  re-baseline procedure + the snapshot drift guard (and the `-m "not
  backtest_regression"` gate / #367).
- `docs/worklog/r1-data-refresh-rebaseline-r1-bloomberg-data-refresh-s27-s32-s34-s35-re-bas.md`
  — R1's "Unresolved / handoff" (the dividends-history drop + provisional S34).
- Memory `backtest-regression-slow-lane-drift` — the #363 serving-logic
  `ev_mean` mechanism + the W16/W30 JPM-window brittleness.
- **Issues:** **#339** (BK↔BNY + CASY + dividends + re-baseline), **#355** (11
  blue-chip backfills), **#354** (PIT fundamentals lookahead), **#357** (W10/W11
  hygiene); **(E)** **#372** (R9→GICS, HIGH), **#369** (#363 IV-gate fallback),
  **#378** (IV-staleness + rate-fallback).
