---
id: S31
title: Compounding-crisis stress (six concurrent downgrade signals)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Every prior Sn entry isolates a single decision-layer
surface (S22 roll defense, S23 event gate + IV-crush, S28 CC
dividend, S29 skew dormancy, S30 HMM regime). S31 asks the
load-bearing follow-up: when **multiple downgrade gates fire on
overlapping candidates at the same `as_of`**, does the engine
produce a coherent output — or do non-composable downgrades
silently fail, missing signals stack in unexpected ways, and the
trader lose the ability to reconstruct *why* a name was filtered?
Wheel-trader pain point: in a real crisis the engine doesn't get
to test one constraint at a time. HMM crisis × dealer short-gamma ×
event lockout × sector concentration × ITM roll defense × CC near
ex-div all hit a live book at once. S31 is the first integration
test that puts six gates on the same anvil.

**Setup.** Anchor date `as_of=2025-04-04` — the AAPL bear → crisis
transition day from S30. Chosen so the HMM stressor fires
**naturally on real-data dynamics**, not as a synthetic config
flag. Provider `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
Account size $250k (institutional sub-PM tier). Universe of 22
names spanning tech (AAPL/MSFT/NVDA/GOOGL/META/TSLA/AMZN/AVGO),
banks (JPM/BAC/WFC/C/GS/MS), defensives (KO/PG/JNJ/VZ/XOM), and
health (UNH/ABBV/LLY).

Existing 5-position book constructed at `BOOK_ENTRY=2025-03-14`
(3 weeks pre-crisis):

- **3 ITM short puts on tech** — AAPL K=250 (spot 188.38, 32.7%
  ITM), MSFT K=420 (spot 359.84, 16.7% ITM), NVDA K=135 (spot
  94.31, 43.1% ITM). Forces `suggest_rolls` into roll-defense
  territory (S22 surface).
- **2 covered calls on banks** — JPM stock basis $240 + call K=250
  (D17 stock-already-owned path), BAC stock basis $45 + call
  K=47 (same shape).

Driver under `%TEMP%\s31\driver.py` (not committed); six
orthogonal checks executed in one run, with
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-c")`
per [[sys-path-worktree-shadow]]. Read-only on decision layer;
positions constructed with `require_ev_authority=False` (the
launch-gate is not the surface under test — see methodology debt).

**Path.** `WheelRunner.rank_candidates_by_ev` at
`engine/wheel_runner.py` (the §2 ranker route — wraps
`EVEngine.evaluate` per candidate). `WheelTracker.suggest_rolls`
at `engine/wheel_tracker.py:2004`. HMM regime layer at
`engine/regime_hmm.py` (output `hmm_regime` + `hmm_multiplier`
into the ranker diagnostic columns). Dealer positioning at
`engine/dealer_positioning.py` (output `dealer_multiplier`,
clamped to `[0.70, 1.05]` per §2). Portfolio-risk gates at
`engine/portfolio_risk_gates.py` (`check_sector_cap`,
`check_kelly_size`, `take_snapshot`). Event surface via
`MarketDataConnector.get_earnings` (per-ticker proxy used in
place of `EventGate.from_bloomberg_calendar` — see F5 for why).

**Status.** Done. **Verdict: the §2 invariants hold under
compounding stress (dealer clamp preserved, no ranker bypass, EV
sign discipline correct), but the engine is operationally
*illegible* in this regime — 20 of 22 candidate names are filtered
silently, the surviving 2 carry no per-name breadcrumb of *why*
the other 20 disappeared, and the trader cannot reconstruct the
gate-by-gate verdict without re-instrumenting the call site.
Five compounding gates verified; the sixth (dealer regime) is
dormant on Bloomberg the same way S29 found for skew.**

**Findings:**

- **(F1 — observability, the killer finding) Universe collapses
  22 → 2 with zero per-name telemetry on which gate filtered
  each name.** With `min_ev_dollars=-1e9` (every numerically-
  computable EV should survive), the ranker returned exactly
  two rows: AVGO ($82.65) and NVDA ($36.22). Twenty other
  names — AAPL, MSFT, GOOGL, META, TSLA, AMZN, JPM, BAC, WFC,
  C, GS, MS, KO, PG, JNJ, VZ, XOM, UNH, ABBV, LLY — were
  filtered upstream and emit *no row* in the output:

  ```
  Returned 2 rows.
  Top 10 by ev_dollars (or all if fewer):
  ticker   spot     iv  premium  ev_dollars  hmm_regime  hmm_multiplier
    AVGO 146.29 0.6415    4.829       82.65     crisis          0.2256
    NVDA  94.31 0.7278    3.592       36.22     crisis          0.2046
  ```

  Plausible filter causes — chain-data unavailable, missing-IV,
  PIT contamination check, event-buffered (3 of the 22 had
  earnings within 7 days), or upstream survivorship — but the
  output gives the trader no way to tell which name failed which
  gate. Same observability shape as S22 F1 (`drops` missing on
  `suggest_rolls`), S28 F2 (`expected_dividend` populating when
  gate blocks), and S29 F5 (`skew_*` columns surfacing when
  dormant). **This is the highest-priority operational finding
  S31 surfaces:** the trader's mental model in a crisis ("show
  me what's safe to trade right now") requires a row per
  *considered* name with a `filter_reason` column, not a
  silent drop. **Logged.**

- **(F2 — positive, §2 verified) HMM crisis multiplier fires
  correctly on the two survivors.** `hmm_regime=crisis` for
  both AVGO and NVDA at `as_of=2025-04-04`, with multipliers
  0.2256 (AVGO) and 0.2046 (NVDA) — both close to the
  documented crisis-state value of 0.2 per
  `engine/regime_hmm.py`. Direction matches S30's confirmed
  AAPL bear → crisis transition on this exact date. The crisis
  multiplier compresses raw EV by ~5×, leaving small absolute
  EV values ($82 / $36) — internally consistent. The engine
  *honestly* downweights even when names survive the
  filter. **Logged as positive.**

- **(F3 — schema gap) The ranker output lacks four columns
  the multi-surface stress test requires.** Driver asked for
  `regime_multiplier`, `event_locked`, `event_reason`, and
  `sector` — none are present in the returned DataFrame:

  ```
  Selected columns present: [ticker, spot, iv, premium, ev_dollars,
    ev_per_day, hmm_regime, hmm_multiplier, dealer_multiplier,
    dealer_regime, skew_multiplier]
  Missing requested columns: [regime_multiplier, event_locked,
    event_reason, sector]
  ```

  - `hmm_multiplier`, `dealer_multiplier`, `skew_multiplier` are
    surfaced separately, but no **combined** `regime_multiplier`
    column shows how they compose into the EV scaling. A trader
    inspecting one row can't tell whether the 0.20 effective
    multiplier came from HMM alone, dealer × HMM, or all three.
  - No `event_locked` / `event_reason` means the silent-filter
    behaviour in F1 is *required* — there's no row to surface
    "blocked: earnings in 7 days" against.
  - No `sector` means portfolio-level reasoning about the ranker
    output requires a separate lookup against `DEFAULT_SECTOR_MAP`
    (F6) — the ranker doesn't help.

  **Logged.** Closing F1 likely requires closing F3 first
  (you need a row per considered name with a filter_reason
  column).

- **(F4 — data, same shape as S29) Dealer multiplier dormant
  on Bloomberg.** Both surviving rows show
  `dealer_multiplier=1.0`, `dealer_regime=None`:

  ```
  Multiplier stats:
    n=2, unique=1, min=1.0000, max=1.0000, mean=1.0000
    in [0.70, 1.05]? True
  ```

  §2 clamp `[0.70, 1.05]` is mathematically preserved (1.0 is
  inside the band), so the test
  `test_dealer_multiplier_evengine_integration.py` (#193) would
  pass here. But the *signal* is dead: no chain access on
  Bloomberg means `engine/dealer_positioning.py` cannot
  compute GEX / gamma flip / walls; the multiplier falls back
  to its safe default of 1.0 and `dealer_regime` returns
  `None`. Same root cause as S29 skew dormancy and
  [[bloomberg-iv-no-skew]]: Bloomberg has no per-strike chain.
  The §2 invariant holds; the *intended functionality* does
  not. **One of the six S31 stressors does not bite on
  Bloomberg; it would bite on Theta. Logged.**

- **(F5 — scenario realism, surprise) The scenario's named
  event-stressor (AVGO earnings) doesn't fire; the actual
  event stressor hiding in the book is JPM.** Per-name
  `get_earnings` probe at `as_of=2025-04-04`:

  ```
  ticker   next earnings   days_out   in_window
  AVGO        2025-06-05         62          no
  JPM         2025-04-11          7         YES  ← held as CC in book
  MS          2025-04-11          7         YES
  WFC         2025-04-11          7         YES
  GS          2025-04-14         10          no
  BAC         2025-04-15         11          no  ← held as CC in book
  C           2025-04-15         11          no
  JNJ         2025-04-15         11          no
  ```

  AVGO (which we chose as the "earnings stressor" target) is
  62 days out — irrelevant to the event gate's typical 5-day
  earnings buffer. The actual hidden stressor is **JPM**: held
  in the book as a covered call, with earnings exactly 7 days
  away — already inside any reasonable event-buffer band. A
  CC sized to expire 28 days out would normally be fine
  through earnings, but the book's risk profile is more
  exposed than the explicit setup acknowledged. **The
  scenario surfaced a real risk the operator didn't plan for;
  this is what stress tests are for.** Logged as a realism
  insight, not a bug.

- **(F6 — sector taxonomy, trader mental-model gap)
  `check_sector_cap` is GICS-strict; "tech" colloquially is
  not "Information Technology" sector.** Probe with the 3-tech
  book loaded and a proposed TSLA short put ($22k notional):

  ```
  TSLA  (Cons Disc)   passed=True   post_open_pct=0.0941
  AVGO  (Info Tech)   passed=False  (existing IT > 25% cap already)
  GOOGL (Comm Svcs)   passed=True   post_open_pct=0.0684
  ```

  AAPL, MSFT, NVDA are all GICS Information Technology; TSLA
  is GICS Consumer Discretionary; GOOGL is GICS Communication
  Services. The driver's check 5 ("compound trade on top of 3
  tech puts") showed `sector_cap.passed=True` for TSLA — *not
  a bug*, but a mismatch with the trader's likely mental model
  ("I'm 60% tech by position count, adding another tech name
  should be blocked"). The colloquial "tech" cluster spans
  three GICS sectors; the gate aggregates per GICS sector
  only. **Logged as a realism gap, not a §2 breach.** Worth
  considering a parallel "growth-cluster" or
  "high-beta-cluster" overlay gate that aggregates across
  the AAPL/MSFT/NVDA/META/TSLA/GOOGL set.

- **(F7 — observability inconsistency) `GateResult.details`
  uses different keys when the gate passes vs. fails.**
  When `check_sector_cap` returns `passed=True`, the details
  bag carries `post_open_sector_pct`. When it returns
  `passed=False`, the same value is carried as
  `sector_pct`. Discovered when the driver's probe used
  `details.get("post_open_sector_pct", 0)` against a failing
  result and silently got `0.0` (the default), masking the
  actual 40+%-of-NAV breach:

  ```python
  if is_allowed:
      return GateResult(details={"sector": ..., "post_open_sector_pct": post_open_pct})
  return GateResult(details={"sector": ..., "sector_pct": post_open_pct, "narrative": ...})
  ```

  (`engine/portfolio_risk_gates.py:319-338`). Caller-side
  code reading details must check both keys, or worse,
  default-to-0 and silently mis-read. Same observability
  shape as F3 schema gaps: shape-dependent on the verdict.
  **Logged — single-line fix, but a real footgun.**

- **(F8 — silent zero, defensive-correct) `suggest_rolls`
  returns zero candidates on all 3 ITM puts with no
  diagnostic of *why*.** Per-position output:

  ```
  AAPL: strike=250.00, current_spot=188.38 (+32.7% ITM)
    suggest_rolls returned ZERO surviving candidates
  MSFT: strike=420.00, current_spot=359.84 (+16.7% ITM)
    suggest_rolls returned ZERO surviving candidates
  NVDA: strike=135.00, current_spot=94.31 (+43.1% ITM)
    suggest_rolls returned ZERO surviving candidates
  ```

  Default `min_net_credit=0.0` keeps credit-only rolls. On
  positions this deep ITM (NVDA 43% ITM, AAPL 33% ITM), no
  credit roll exists in the (DTE × delta) grid; every
  candidate would require a debit. The honest output ("no
  roll") is the right call — this is **defensively correct**:
  the engine refuses to give bad advice (which is what S22
  F2 chastised the pre-fix tracker for). But the trader
  receiving zero rows can't tell:
  - Did all rolls have negative EV after the crisis multiplier?
  - Did all rolls fail the `min_net_credit=0` filter
    (rescue-debit could still be sensible)?
  - Did all rolls fail because the underlying moved past the
    strike-solver's safe band?

  An attached `reason_tally` (e.g. `{"all_debit": 12,
  "ev_negative": 8, "moneyness_floor": 0}`) would let the
  trader decide between "close and take assignment" vs
  "rescue-debit roll." Same observability pattern as F1 and
  F3. **Logged.**

- **(F9 — observability, combined-multiplier omission;
  surfaced during V1 verification) The combined
  `regime_multiplier` (HMM × skew × news × credit) is NOT
  surfaced in the ranker output, even though it is the single
  value that scales `ev_raw → ev_dollars`.** The components
  `hmm_multiplier`, `skew_multiplier`, `news_multiplier`,
  `credit_multiplier` ARE in the output (46-column DataFrame),
  but no combined `regime_multiplier` column exists. A trader
  who doesn't know `credit_multiplier` exists (it's not in
  any "obvious" diagnostic subset and was not in the F3
  list) will compute `ev_raw × hmm × skew × news` and find a
  20% gap with `ev_dollars` on a crisis day — no
  explanation. The author of this Sn entry made exactly this
  mistake on first read; the V1 verification surfaced the
  discrepancy and traced it to `credit_mult = 0.80` (line
  `engine/wheel_runner.py:773`, fires when global
  `credit_regime == "crisis"`). Same observability family as
  F1 / F3. **Logged.**

- **§2 verified (with stress-coverage caveats).** Across all
  six checks: (a) the dealer multiplier never moved outside
  `[0.70, 1.05]` — but was always 1.0 on Bloomberg
  (dormant), so the clamp was preserved **vacuously, not
  under stress**. The first-PR framing "§2 invariants
  survive compounding stress intact" overstated this — the
  clamp was never put under stress on this provider. (b) Every
  ranker-output row was generated via `EVEngine.evaluate`,
  and per V5 the 20 silently-filtered names are blocked by
  `event_gate`'s holding-window check, NOT by a non-EV
  rescue path. (c) `suggest_rolls` returning zero candidates
  is the conservative refusal per V6 + V7 — confirmed
  defensive, not bypass. **The §2 contract holds; the
  stress only meaningfully tested some of it.** Logged with
  the corrected scope.

**Verification probe (V1–V7).** A follow-up driver
(`%TEMP%\s31\verify.py`) closed the gap between *engine output
observed* and *engine output verified* for the most uncertain
claims above. The first-PR framing was honest about what was
observed but loose about what was *verified*; this section
upgrades or downgrades each claim accordingly:

- **V1 — composition arithmetic VERIFIED CORRECT.** For both
  survivors, the full multiplier chain matches `ev_dollars`
  to better than one cent:

  ```
  AVGO: ev_raw=457.89, hmm=0.2256, skew=1.0, news=1.0, credit=0.80
        product           = 457.89 × 0.2256 × 1.0 × 1.0 × 0.80 = 82.66
        ev_dollars output = 82.65  ✓ (rounding)

  NVDA: ev_raw=221.32, hmm=0.2046, skew=1.0, news=1.0, credit=0.80
        product           = 221.32 × 0.2046 × 1.0 × 1.0 × 0.80 = 36.23
        ev_dollars output = 36.22  ✓ (rounding)
  ```

  The arithmetic is exact. The previously-undocumented
  factor was `credit_mult = 0.80` from
  `engine/wheel_runner.py:773` (fires when global
  `credit_regime == "crisis"` per `fa.credit_regime`).

  **2026-05-28 post-#260+#262 environment footnote (S46).** Re-running
  the V1 probe on the Cowork sandbox at `origin/main` @ `56d8e5c`
  returns `credit_multiplier = 1.0` (not 0.80) on this date because
  `FREDAdapter.credit_regime` raises `TypeError` on the empty
  HY OAS series and `wheel_runner.py:870`'s broad
  `except Exception: credit_mult = 1.0` silently absorbs it.
  Identity still holds with the new factor value
  (AVGO: 457.89 × 0.2256 × 1.0 × 1.0 × 1.0 = 103.30,
  observed `ev_dollars = 103.31`). The drift is data-environment-
  driven (FRED unavailable here), NOT an engine code change —
  `engine/wheel_runner.py` is git-unchanged on the credit-regime
  branch since 2026-05-24. Filed as `S46` §5 F1 + F2.

- **V2 — BSM fair-value sanity VERIFIED.** Both rows have
  `premium == fair_value` exactly (AVGO 4.829, NVDA 3.592)
  and `edge_vs_fair = 0`. Confirms the Bloomberg path
  (premium is BSM-derived, not quoted) per S1 / S29 F1.
  Same finding, fresh verification.

- **V3 — HMM state-probability decomposition VERIFIED.**
  Fitting the same 504-day-tail HMM directly on AVGO and
  NVDA log-returns at `as_of=2025-04-04`:

  ```
  AVGO state_probs   = [crisis 0.9680, bear 0.0000, normal 0.0320, bull_quiet 0.0000]
       Σ probs × wts = 0.9680*0.2 + 0*0.5 + 0.0320*1.0 + 0*1.25 = 0.2256  ✓
       matches ranker hmm_multiplier (0.2256) within 0.000017.

  NVDA state_probs   = [crisis 0.9941, bear 0.0003, normal 0.0056, bull_quiet 0.0000]
       Σ probs × wts = 0.2046                                         ✓
       matches ranker hmm_multiplier (0.2046) within 0.000026.
  ```

  Both names are 96–99% pure crisis state. The first-PR
  F2 framing "close to 0.20" was lazy — the math is *exact*,
  the values are probability-weighted blends of the four
  state multipliers, and the blend is dominated by crisis on
  this date. **F2 upgrade: verified mathematically clean.**

- **V4 — missing-data categorization of the 20 filtered
  names: ZERO data-layer drops.** Per-name probe: all 20 had
  OHLCV at `as_of`, all 20 had PIT IV (both `hist_put_imp_vol`
  and `hist_call_imp_vol`), and 0 had earnings within the
  default 5-day buffer.  So the silent filter is NOT caused
  by missing data, missing IV, or near-term-earnings (within
  documented default `earnings_buffer_days=5`).

- **V5 — the silent filter IS `event_gate` firing on the
  holding-window earnings overlap.** Re-running
  `rank_candidates_by_ev` with `use_event_gate=False`
  surfaced **all 20 missing names** with positive EV. The
  newly-surfaced rows have EVs ranging from $5.50 (VZ) to
  $720.48 (LLY); 17 are in `hmm_regime=crisis`, one in
  `bear`. Earnings for the 7 blocked-but-not-≤5-days names
  (JPM, MS, WFC at +7d; GS at +10d; BAC, C, JNJ at +11d;
  UNH at +13d) all sit INSIDE the 35-DTE holding window of
  the proposed trade. The `event_gate` blocks any trade
  whose holding window brackets an earnings event — that's
  exactly the right behaviour. **F1's "killer observability"
  finding upgrades to:** the engine is acting correctly,
  but the output offers the trader no way to see *why* each
  name was filtered. The decision is right; the explanation
  is missing.

- **V6 — `suggest_rolls` returns 0 candidates across 4 grid
  shapes on 43%-ITM NVDA**, including longer DTE (180d),
  debit allowed (`min_net_credit=-$500`), and very far OTM
  (5-delta strikes). The honest output ("no defensible
  roll") survives every parameter widening tested.

- **V7 — `suggest_rolls` returns 1 candidate on 0.9% ITM
  AAPL (control).** Engine WORKS on shallow ITM. The AAPL
  candidate carries `net_credit_debit = +$23.39` (passed
  the credit filter) but `roll_ev = −$141.28` (negative EV
  surfaced because the credit-only filter accepted it).
  **V6's zero on 43% ITM is therefore *defensible refusal
  at extreme moneyness*, not a bug.** F8 downgrade: the
  silent zero is correct in conclusion; the observability
  gap (no `reason_tally` of which (DTE, delta) cells failed
  and why) remains the only operational issue.

**Realism Check.**

| Aspect | Engine (Bloomberg, 2025-04-04) | Real-market / sound-trader expectation | Verdict |
|---|---|---|---|
| 22-name universe ranker scan | 2 rows returned, 20 silently filtered. **V5 proves filter is `event_gate` firing on 35-DTE holding-window earnings overlap.** All 20 missing names surface with positive EV when `use_event_gate=False` | A real trader at a desk would expect either a row per considered name (with `filter_reason`) or a header summary of "20 skipped — N event-blocked, M missing-IV, ..." | ⚠ Decision correct (event_gate fires on real overlap); operability gap remains (silent drop) |
| HMM regime on a confirmed crisis day | `hmm_regime=crisis`, multipliers 0.2256 / 0.2046. **V3 verifies state_probs decomposition: AVGO 96.8% crisis + 3.2% normal; NVDA 99.4% crisis + 0.06% bear + 0.56% normal — products match to within 3e-5** | S30 already confirmed this transition on AAPL log-returns; VIX > 30 in April 2025; "crisis" is the right verdict | ✓ Verified mathematically |
| Crisis multiplier × raw EV composition | **V1 verifies** `ev_dollars = ev_raw × hmm × skew × news × credit_mult` exactly: AVGO 457.89 × 0.2256 × 1.0 × 1.0 × 0.80 = 82.66 ≈ 82.65 ✓; NVDA 221.32 × 0.2046 × 1.0 × 1.0 × 0.80 = 36.23 ≈ 36.22 ✓ | Engine arithmetic should be reproducible from surfaced components — the 5th factor (credit_mult) was the missing piece | ✓ Verified to <1¢ tolerance |
| Dealer multiplier under compounding stress | 1.0 across both survivors, `dealer_regime=None` | Real crisis days have *meaningful* dealer-positioning signal (forced unwinds, short-gamma below put walls) | ⚠ Dormant (Bloomberg-only, same shape as S29 skew) |
| §2 clamp `[0.70, 1.05]` on dealer multiplier | always 1.0 → trivially in band; **stress was not applied on this provider — clamp not exercised under load** | The clamp pinned by `test_dealer_multiplier_evengine_integration.py` (#193) | ⚠ Vacuously preserved (clamp not stressed; would need Theta replay to test under load) |
| Per-name event-gate visibility (output schema) | No `event_locked` / `event_reason` columns in ranker output; `event_gate` correctly fires per V5 but the trader cannot see why | Trader needs to know which book positions hit earnings windows; especially when a CC's expiry brackets earnings | ❌ Column missing (decision correct, telemetry absent) |
| Combined `regime_multiplier` in output | **F9 — missing entirely.** Components (`hmm_multiplier`, `skew_multiplier`, `news_multiplier`, `credit_multiplier`) are present; the trader must multiply 4 columns to reconstruct the EV scaling | Output should expose the actual scalar that multiplies `ev_raw → ev_dollars`, so verifying the composition doesn't require knowing the 4-factor formula | ❌ Column missing |
| Sector concentration with 3 IT puts + new TSLA | `sector_cap.passed=True` (TSLA = Consumer Discretionary, not IT) | Trader's mental model: "60% tech by count → block another tech name." Engine's: "TSLA ≠ IT → 9% Cons Disc, allowed." Both technically correct; mental models diverge | ⚠ Realism mismatch (taxonomy) |
| `GateResult.details` key consistency | `post_open_sector_pct` (pass) vs `sector_pct` (fail) | Caller-side observability should be shape-stable: same key, regardless of verdict | ❌ Key asymmetry (footgun) |
| `suggest_rolls` on 43% ITM puts | Returns 0 candidates across 4 grid shapes (V6); **V7 confirms 1 candidate returned on 0.9% ITM control — engine WORKS on shallow ITM, V6's zero is defensible refusal at extreme moneyness via the `min_net_credit=0` filter killing all-debit candidates** | A trader needs to choose between close-and-take-assignment, rescue-debit roll, or hold. Zero candidates with no `reason_tally` doesn't help that decision | ⚠ Decision defensible (correct refusal); operability gap remains |
| BSM fair-value sanity on the Bloomberg path | **V2 verifies** `premium == fair_value` on both rows (AVGO 4.829 / NVDA 3.592), `edge_vs_fair = 0` | Bloomberg path is BSM-derived per S29 F1 / S1 (no quoted chain) | ✓ Verified |

**Verdict.**

- **Engine math is verified correct under stress; the §2
  invariants hold to the extent the provider lets them be
  stressed.** V1 verifies the 5-factor composition arithmetic
  to <1¢ tolerance: `ev_dollars = ev_raw × hmm × skew × news ×
  credit_mult` exactly. V3 verifies the HMM
  state-probability decomposition is mathematically clean
  (matches to 3e-5). V2 confirms BSM fair-value sanity on
  the Bloomberg path. V5 confirms the silent filter of 20
  names is `event_gate` correctly firing on the 35-DTE
  holding-window earnings overlap (NOT random drops). V7
  confirms `suggest_rolls` works on shallow ITM and refuses
  appropriately at extreme moneyness. **The decision-layer
  arithmetic, regime detection, and gate-firing logic are
  all verified to behave as documented.** The recent
  campaign-close tests (`test_evengine_event_lockout.py`,
  `test_dealer_multiplier_evengine_integration.py`,
  `test_f4_tail_risk_gap.py`,
  `test_consume_ranker_row_anchor.py` — PRs #185 / #186 /
  #193 / #196) all hold up under this composed stress.

- **§2 framing must be honest about the stress that was
  applied.** The dealer-multiplier clamp `[0.70, 1.05]`
  was preserved trivially (always 1.0 on Bloomberg per F4
  — dealer module dormant for chain-access reasons). The
  clamp was NOT exercised under load on this provider; the
  test that pins it (`test_dealer_multiplier_evengine_integration.py`
  / PR #193) holds, but S31 does not add fresh stress
  evidence to the clamp. A Theta replay (S6) would put the
  clamp under genuine load on a crisis day. The first-PR
  framing "§2 invariants survive compounding stress
  intact" was thus over-claimed; the corrected scope is
  "§2 invariants are not breached; the dealer clamp was not
  stressable on this provider."

- **The engine is operationally *illegible* in this
  regime, but the underlying decisions are defensible.**
  F1 (silent filter), F3 (schema gaps), F5 (hidden book
  stressor), F7 (key asymmetry), F8 (silent-zero roll),
  and F9 (combined-multiplier omission) all share one
  shape: *the engine has the right answer, the output does
  not let the trader see why*. V4 + V5 prove the silent
  filter is correctly-acting `event_gate`. V6 + V7 prove
  the silent-zero roll is correctly-acting refusal. The
  engine is structurally sound and decisionally
  defensible; the trader-facing surface is the gap. This
  is the same observability pattern S22, S28, S29
  surfaced — but S31 is the first time it has been
  documented as a composed pattern across the whole
  decision layer in one cycle.

- **The dealer dormancy on Bloomberg is now a confirmed
  multi-Sn pattern, not an isolated bug.** S29 found
  skew dormant on Bloomberg for the same root cause (no
  per-strike chain). S31 confirms the dealer module sits
  in the same condition. Both modules are well-built and
  unreachable on the live provider. Closing this gap is
  a connector-migration problem, not an engine-code
  problem — same conclusion as [[bloomberg-iv-no-skew]]
  and [[bloomberg-data-refresh-blocked]].

- **F5 (the JPM CC earnings overlap) is the kind of
  insight stress tests *should* surface.** The scenario
  was designed with AVGO as the "earnings stressor"; the
  data revealed JPM held in the existing book actually
  hits the earnings window. A real trader running this
  book at this date would care; the engine had the data
  to flag it (`get_earnings`) but didn't surface it in
  any of the ranker / tracker outputs. The trader has to
  ask three different APIs to assemble the picture.

**AI handoff.**

**✓ Shipped status (post-campaign, 2026-05-25/26):** All six
Fix# entries below shipped against the findings they
referenced. The bullets are retained for historical rationale;
the actual code lives in the PRs listed here.

| Fix # | Targets | Shipped in |
|---|---|---|
| Fix #1 + Fix #4 | F1 silent filter, F4 / F8 suggest_rolls silent zero | **PR #209** — `fix(decision-layer): add drops_summary attribute to all ranker / roll-suggest frames`. **Discovery during verification:** the per-candidate reason list (`df.attrs["drops"]`) was *already* populated on every ranker / suggest_rolls return path (S22 F1 fix shipped earlier); the actual gap was *discoverability*. PR #209 adds the trader-facing `df.attrs["drops_summary"] = {"total_dropped": N, "by_gate": {...}}` roll-up on all 6 ranker returns + both suggest_rolls + suggest_call_rolls returns, applied uniformly via a module-level `_attach_drops_summary` helper. Closes F1 / F4 / F8 as composed-pattern observability — the engine had the right answer; the trader can now see *why* a name was dropped at a glance. |
| Fix #2 | F3 schema gap (the `sector` slice) | **PR #210** — `fix(wheel_runner): surface GICS sector column on all ranker survivor rows`. Adds a CORE `sector` column (sourced from `DEFAULT_SECTOR_MAP`, same map `check_sector_cap` aggregates by) to all 3 ranker outputs. F3's other requested columns (`regime_multiplier`, `event_locked` / `event_reason`) are addressed by **PR #208** (regime_multiplier) and PR #209's `drops_summary` (event_locked / event_reason now inferable from `by_gate` since `event_gate` is the documented `event` gate). |
| Fix #3 | F7 GateResult key asymmetry | **PR #207** — `fix(portfolio_risk_gates): rename check_sector_cap fail-path details key for pass/fail symmetry`. One-line rename `sector_pct` → `post_open_sector_pct` in the failure-path details dict + 2 test updates + regression test pinning the symmetry. |
| Fix #5 | F6 alt-cluster overlay (the GICS-strict mental-model gap) | **PR #212** — `feat(portfolio_risk_gates): add check_alt_cluster_cap for colloquial-cluster exposure`. Complementary to `check_sector_cap`; aggregates across colloquial clusters (default ships `mega_cap_growth = {AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AVGO}`); operators pass `clusters` kwarg for custom definitions. Default cap 0.40 (looser than the 0.25 sector cap). Standalone gate — not wired into the D17 hard-block flow yet (follow-up: opt-in mechanism). |
| Fix #6 | F9 combined-multiplier omission | **PR #208** — `fix(wheel_runner): surface combined regime_multiplier column in ranker output`. Adds `regime_multiplier` column (= `res.regime_multiplier`, the engine's *final* multiplier post-clamp + heavy_tail_penalty + dealer_mult — the scalar that actually scales `ev_raw → ev_dollars`). Regression test pins the composition identity. |

**What was NOT shipped:** the original Fix #1 framing (add
`include_filtered_candidates=True` + `filter_reason` per dropped
candidate as new rows in the output) — superseded by the V5
verification discovery that `df.attrs["drops"]` was already
populated, making the rows-only addition redundant. The
discoverability fix (drops_summary on top of the existing drops
list) achieves the same trader operability with less surface
area. Documented for posterity.

The original Fix# bullets that follow are kept for the design
rationale; replace the "do this" framing with "*was done in
PR #YYY*" when reading them.

---

- **Fix #1 (highest-priority, the F1 closer):
  `filter_reason` per dropped candidate in
  `rank_candidates_by_ev`.** Currently the ranker returns
  only surviving rows; modify to optionally return a
  row per *considered* candidate with `filter_reason ∈
  {"chain_unavailable", "missing_iv", "event_blocked",
  "ev_negative", "pit_violation", ...}` and either
  `ev_dollars=NaN` or a separate `passed` boolean. Gate
  behind an `include_filtered_candidates=True` kwarg to
  preserve backwards compatibility. This unblocks the
  whole "trader observability" story F1/F3/F5/F8 all
  point at. Smallest-scope fix; biggest operational
  impact.

- **Fix #2 (F3 column additions): surface a combined
  `regime_multiplier`, `event_locked`, `event_reason`,
  `sector` per ranker row.** `regime_multiplier`
  composes from `hmm_multiplier × dealer_multiplier ×
  skew_multiplier × event_multiplier` (or whichever
  set is live). `event_*` come from the event_gate
  hit. `sector` comes from `DEFAULT_SECTOR_MAP`.
  Pairs with Fix #1 — once `include_filtered_candidates`
  surfaces dropped rows, these columns let the trader
  see which gate did the dropping.

- **Fix #3 (F7 key asymmetry):** in
  `engine/portfolio_risk_gates.py:319-338`, use
  `post_open_sector_pct` in *both* pass and fail
  details dicts. One-line change; eliminates the silent
  default-to-zero footgun. Cheap.

- **Fix #4 (F8 reason_tally for suggest_rolls):**
  modify `WheelTracker.suggest_rolls` to attach a
  `_reason_tally` attribute (or return tuple) when zero
  surviving candidates: which (DTE, delta) pairs failed
  EV, which failed `min_net_credit`, which hit the
  moneyness floor. Lets the trader distinguish "close
  the position" from "rescue-debit roll" from "hold."

- **Fix #5 (F6 alternative-clustering overlay):**
  Optional. Add a parallel "growth-cluster" /
  "high-beta-cluster" exposure check alongside the
  GICS sector cap, so a trader with the
  AAPL/MSFT/NVDA/META/TSLA/GOOGL mental model gets
  warned even when GICS doesn't aggregate them.
  Lower priority — this is realism polish, not a
  correctness gap.

- **Fix #6 (F9 combined-multiplier surfacing):** add a
  `regime_multiplier` column to the ranker output, equal
  to `hmm_multiplier × skew_multiplier × news_multiplier
  × credit_multiplier` (the exact product passed into
  `EVEngine.evaluate`). The components are already in
  the 46-column DataFrame; adding the product is a
  one-line computation in the ranker. Closes the F9
  observability gap that the verification driver surfaced
  (the entry author themselves got tripped up on first
  read). Cheap; pairs naturally with Fix #1.

- **Sanity follow-up Sn (S6 dependency): re-run S31 on a
  Theta replay** to confirm the dealer multiplier
  *actually moves* below 1.0 on a real crisis day with
  chain access (NVDA short-gamma below put wall in April
  2025 is the expected regime). Would close the
  "dormancy is data not logic" hypothesis the same way
  S29 → S6 closes it for skew.

- **Test promotion (per the two-axis verification
  pattern in `TESTING.md`):** F1's silent-filter
  contract — that `rank_candidates_by_ev` either
  surfaces a row per considered candidate or
  documents the drop somewhere queryable — is the
  right shape for a pytest regression test (likely as
  an xfail today, flipping to pass when Fix #1
  ships). Mirrors the
  `tests/test_f4_tail_risk_gap.py` xfail-as-watchlist
  pattern from PR #196.

**Methodology debt.**

- **Single connector (Bloomberg).** The dormancy finding
  (F4) applies to Bloomberg only; Theta would activate
  the dealer module. S6 (queued) would let us cleanly
  separate "implementation dormant" from "data dormant"
  for dealer the way S29 separates it for skew.

- **Single as_of.** S31 tests one crisis day. The
  composed-multiplier behaviour on a normal-quiet day
  (e.g. 2026-01-15) versus a bear-but-not-crisis day
  (e.g. 2025-04-11) would either confirm "the silent
  filter is regime-invariant" or surface
  "crisis-day-specific filter behaviour" — useful in
  either direction.

- **Synthetic book.** The 5 positions were constructed
  (strikes picked to be ITM, entry dates fabricated 3
  weeks before crisis). A real audit-trail book from
  the same date would surface different latent
  stressors — F5 (the JPM CC hit) is illustrative of
  what real-book audits surface that synthetic books
  miss.

- **`require_ev_authority=False`.** The driver
  bypassed the launch-gate to construct positions
  directly. This was deliberate (the launch-gate is
  not the surface under test), but it means the
  D17 portfolio-risk hard-blocks (sector cap, Kelly,
  delta) were not exercised through the
  position-opening path. F6's sector finding was
  verified via direct `check_sector_cap` calls; the
  full integration through `open_short_put`'s
  hard-block path is `test_authority_hardening.py`
  territory and was not re-tested here.

- **No comparison to Theta replay on the same date.**
  Pairs with the "Sanity follow-up Sn" above. On a
  date with real chain data, the dealer module
  *should* meaningfully move the multiplier on the
  NVDA-short-gamma-below-put-wall pattern that was
  certainly present in April 2025.

- **`suggest_call_rolls` not tested.** The driver
  exercised `suggest_rolls` on the 3 ITM tech puts
  but did not run `suggest_call_rolls` on the JPM/BAC
  CCs near their earnings (F5). The dividend-near
  call-roll path (S28 F3 surface) was not stressed
  here; would pair naturally with a follow-up Sn.

---
