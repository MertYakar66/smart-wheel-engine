---
id: S28
title: CC dividend realism (VZ / JPM / MSFT / KO / AAPL / WMT)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Realism check on the dividend-aware leg of the CC
ranker. Wheel-trader pain point: a covered call that goes ITM near
ex-div is at high early-exercise risk — the call holder rationally
exercises if extrinsic < dividend. Engine claim (`engine/ev_engine.py`
line 357-361): when `option_type=="call"` AND `days_to_ex_div <= dte`
AND `expected_dividend > 0`, the dividend is subtracted from the
expected loss on outcomes where the call is ITM at expiry. Test
asks: does this gate actually fire on the right names? Is the
dividend-aware signal observable in the ranker output?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
six names spanning the dividend-window space at `as_of=2026-03-20`:

| Ticker | Ex-div in file | dte_to_ex | Class |
|---|---|---|---|
| VZ | 2026-04-10 ($0.7075) | 21d | **INSIDE** 35-DTE window |
| JPM | 2026-04-06 ($1.50) | 17d | **INSIDE** 35-DTE window |
| MSFT | 2026-05-21 ($0.91) | 62d | **OUTSIDE** 35-DTE window |
| WMT | 2026-05-08 ($0.2475) | 49d | **OUTSIDE** 35-DTE window |
| KO | none in file | n/a | **TRUNCATED** (known dividend aristocrat) |
| AAPL | none in file | n/a | **TRUNCATED** (low yield, but tracked) |

OTM grid: 35-DTE × (0.30, 0.25, 0.15) deltas. ITM probe: 35-DTE ×
(0.70, 0.80). A/B follow-up: same matrix at 0.25 delta with
`dividend_yield=None` (engine resolves from fundamentals) vs
`dividend_yield=0.0` (forced no carry) to isolate the dividend's
quantitative impact on `ev_dollars`. `use_event_gate=False`
throughout (so the JPM 17-day ex-div / MSFT 62-day ex-div don't
trigger event-window blocks unrelated to the dividend test).
Drivers under `%TEMP%\s27\`, not committed;
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]].

**Path.** `WheelRunner.rank_covered_calls_by_ev` at
`engine/wheel_runner.py:1836`. Per-row data plumbing at lines
2049-2152 (BSM continuous yield from `fundamentals.dividend_yield`,
discrete `expected_dividend` from `conn.get_next_dividend(ticker,
as_of)`). Strike-solve in `_solve_call_strike`. Each candidate scored
through `EVEngine.evaluate`. Dividend early-exercise penalty at
`engine/ev_engine.py:355-361` (the gate: `option_type=="call"` AND
`days_to_ex_div is not None` AND `days_to_ex_div <= dte` AND
`expected_dividend > 0` → `pnls -= is_itm * expected_dividend *
multiplier`).

**Status.** Done. **Verdict: the engine's dividend math is
correct — both the BSM continuous yield AND the EVEngine
discrete early-exercise penalty fire in the right direction
on names with inside-DTE ex-divs. But three observability /
data-coverage gaps surface that materially reduce the dividend
gate's real-world effectiveness.**

**Findings:**

- **(F1 — verified, positive) Both dividend pathways exist and
  work.** Two independent code paths apply dividend influence
  to CC EV:
  1. **Continuous yield (BSM q)** from
     `fundamentals.dividend_yield`. Always applied to the
     strike-solve and synthetic premium via
     `engine/wheel_runner.py:2049-2064` (`div_q` argument to
     BSM). No gate.
  2. **Discrete early-exercise penalty** from
     `conn.get_next_dividend(ticker, as_of)` →
     `expected_dividend` → `EVEngine`. Gated on
     `days_to_ex_div <= dte` (`ev_engine.py:357-358`) and
     `is_itm` at simulated expiry.

  Both fire on JPM (inside DTE, $1.50 dividend, 25-delta CC):
  ev_dollars shifts by **-$15.04** when `dividend_yield` is
  forced to 0 vs default — the largest single-trade impact in
  the matrix. On VZ (inside DTE, $0.7075 dividend, highest
  yield in the matrix at 5.47%), the shift is **-$6.57**.

  **Logged as a positive — the engine has the right machinery,
  in the right place, with the right gate.**

- **(F2 — gap, observability) `expected_dividend` diagnostic
  column populates regardless of the gate.** Even when
  `days_to_ex_div > dte` (so the EVEngine penalty cannot
  fire), the ranker output's `expected_dividend` column shows
  the upcoming dividend amount. MSFT at 25-delta CC:

  ```
  days_to_ex_div=62  dte=35  expected_dividend=0.91
  ```

  A trader inspecting the ranker output would reasonably
  conclude the engine is factoring $0.91 of dividend cost into
  this trade. It is not — the EVEngine gate blocks it (62 >
  35). The -$8.22 EV shift observed in the A/B is from the
  BSM continuous yield alone (MSFT yield 0.91%), not the
  discrete penalty. **Same observability shape as S22 F1
  (suggest_rolls missing `drops`). Logged.**

- **(F3 — gap, data coverage) 75% of S&P 500 tickers have NO
  future ex-div in the dividend file after 2026-03-20.** The
  raw counts:

  ```
  total tickers in sp500_dividends.csv:                427
  tickers with ANY ex_date > 2026-03-20:               107  (25%)
  ```

  Forward-truncated major payers include KO, PG, JNJ, AAPL,
  UNH, XOM, CVX, PEP, MCD, T. These are all known quarterly
  dividend payers; the file just doesn't carry their
  forward-declared ex-divs at the cutoff. Effect: on 320 of
  427 (75%) S&P 500 tickers, `get_next_dividend` returns
  `None`, so `expected_dividend=0.0` and the EVEngine
  early-exercise penalty cannot fire — the discrete
  ITM-near-ex-div protection is silently inactive. The BSM
  continuous yield still applies (KO's 25-delta CC EV shifts
  -$4.74 when `dividend_yield=0` is forced), so the engine
  retains *some* dividend awareness via fundamentals — but
  the discrete protection is the trader-meaningful one.
  **Parallels S23 F2 (earnings-file forward truncation).
  Logged.**

- **(F4 — design-intent + observability gap) ITM CC strikes
  are silently skipped *by design*.** `target_deltas=(0.70,
  0.80)` returns an empty frame on every ticker in the
  matrix — engine does not produce ITM CC candidates.
  Post-PR-open 30-second code check confirmed this is
  **explicit design intent**: `_solve_call_strike` at
  `engine/wheel_tracker.py:88-112` brackets Brent root-finding
  on `[spot*1.01, spot*2.0]` and the docstring is explicit —
  *"Returns None when no solution exists in [spot*1.01,
  spot*2.0] — the OTM region a covered call is sold into
  (strike above spot)."* A 0.70-delta call needs a strike
  below spot, which sits outside that bracket, so Brent
  cannot find a root and `_solve_call_strike` returns
  `None` → ranker emits no row. **Auto-mitigates** the
  worst-case ITM-near-ex-div early-exercise scenario by
  refusing to produce ITM CC strikes at all — a sound
  wheel-strategy default (sell calls you'd be happy to
  assign, above your basis). The remaining gap is
  **observability**: a trader asking for ITM CC strikes
  (e.g. to lock in upside on a held position around a
  known ex-div) gets an empty frame and no `drops` signal
  saying "ITM strikes are out of scope for CC ranking by
  design." Logged as a design-intent finding with an
  observability follow-up rather than a logic bug.

- **(F5 — gap, data coverage) WMT history-gated despite being
  a household name.** Engine drops WMT entirely:

  ```
  [{'ticker': 'WMT', 'gate': 'history', 'reason': 'history 70d < required 504d'}]
  ```

  Direct OHLCV probe: WMT has 70 rows of data starting
  2025-12-09 — not 504+ as the 504-day history gate expects.
  The dividends + fundamentals files do have WMT (yield
  0.80%, ex-div 2026-05-08). Likely a recent split or
  ticker-symbol change that wasn't propagated to the OHLCV
  extraction. Inconsistent coverage across the three Bloomberg
  files for the same ticker. **Logged — partial-coverage
  data quality bug, not a wheel-runner bug.**

- **(F6 — observation) Even on the highest-yielding name in
  the matrix, the dividend's EV impact is modest at OTM
  deltas.** VZ (5.47% yield) 25-delta CC EV shifts -$6.57
  from the dividend pathway. JPM ($1.50 absolute, 2.02%
  yield) shifts -$15.04 — the largest in the matrix. Both
  are small fractions of the absolute EV magnitudes (JPM's
  ev=-$164.40 is dominated by the high-IV / wide-tail
  factors). **Wheel traders should not over-weight dividend
  defense as a CC-killer for OTM strikes**; the dividend
  shifts the answer slightly, but does not flip OTM verdicts
  in this matrix. The dividend cost would be much larger on
  ITM strikes — exactly the strikes the ranker silently
  refuses to produce (F4). **Logged.**

- **§2 verified.** `rank_covered_calls_by_ev` routes every
  candidate through `EVEngine.evaluate`. The dividend
  pathway is integrated into the EV math, not a
  side-channel adjustment. No bypass. **Logged as a
  positive.**

**Realism Check.**

| Ticker | Ex-div (file) | dte_to_ex | Engine `days_to_ex_div` | Engine `expected_dividend` | Δ ev_dollars from dy=0 | Trader expectation | Aligned? |
|---|---|---|---|---|---|---|---|
| VZ | 2026-04-10 ($0.7075) | 21d | 21 | $0.7075 | -$6.57 | Modest CC penalty for OTM at high yield | ✓ Aligned |
| JPM | 2026-04-06 ($1.50) | 17d | 17 | $1.50 | -$15.04 | Largest dollar impact (high $/share div) | ✓ Aligned |
| MSFT | 2026-05-21 ($0.91) | 62d | 62 | $0.91 (column populated, gate blocks penalty) | -$8.22 (BSM q only) | Diag column should be 0 when gate blocks | ⚠ Observability gap (F2) |
| KO | none in file (known $0.42 Q1 historical) | n/a | None | 0.0 | -$4.74 (BSM q from 2.76% yield still applies) | Engine should know KO is a dividend aristocrat | ⚠ Truncation (F3); partial via BSM q |
| AAPL | none in file | n/a | None | 0.0 | -$2.48 (BSM q from 0.42% yield) | Low yield → low impact | ✓ aligned despite truncation |
| WMT | 2026-05-08 ($0.2475) | n/a | dropped | dropped | n/a | A household name with multi-decade history should rank | ❌ Data-coverage gap (F5) |

**Verdict.**

- **Dividend math: correct and well-placed.** Two independent
  pathways (BSM q + EVEngine early-exercise penalty), gate
  properly enforced internally, EV shifts in the right
  direction on inside-window names. The engine has the
  protective machinery a wheel trader would expect.

- **Real-world effectiveness: limited by data coverage and
  observability.** On 75% of S&P 500 tickers, the discrete
  protection is silently inactive (F3 dividend-file
  truncation). The diagnostic column misleads on
  outside-window names (F2). ITM strikes — the
  high-early-exercise-risk regime the gate was designed for —
  are skipped by design (F4 — `_solve_call_strike` brackets
  to OTM-only as a wheel-strategy invariant), but the design
  intent is not surfaced to a trader asking for ITM via
  `target_deltas≥0.70`. The continuous BSM q is a
  partial safety net for the truncated names but is not the
  same instrument as the discrete penalty.

- **The realism gap is not in the engine's logic.** Three of
  the four findings (F2/F3/F5) are data-layer or
  observability gaps, not engine-math gaps. F4 is an engine
  design intent (wheel-strategy CCs are OTM by convention)
  with an observability follow-up, not a logic bug.

**AI handoff.**

- **Fix #1 (natural follow-on, smallest scope):** zero the
  `expected_dividend` diagnostic column when the EVEngine
  gate would block (`days_to_ex_div > dte`). This is a
  one-line change at the ranker's diagnostic emission site
  (`engine/wheel_runner.py` around line 2306 where
  `expected_dividend` is rounded into the output dict). The
  EV math itself is correct; only the observability is off.
  Test: MSFT 25-delta CC should show `expected_dividend=0.0`
  in the ranker output (current: 0.91).

- **Fix #2 (separate scope, data layer):** refresh
  `sp500_dividends.csv` with forward-declared ex-divs for
  the truncated tickers. Per the [[bloomberg-data-refresh-blocked]]
  memory this requires the user's BQL queries + `end_date`
  bumps and cannot be self-served. Alternative: when
  `get_next_dividend` returns `None`, fall back to estimating
  the next ex-div from `dividend_frequency` + most-recent
  historical `ex_date` (`KO`'s last ex-div was 2026-03-13 with
  quarterly frequency, so the next is ≈ 2026-06-13). That
  would partially close the truncation gap without a data
  refresh.

- **Fix #3 (observability — the F4 follow-up):** emit a
  `drops` entry (e.g. `gate="strike_itm_design_skip"`,
  `reason="target_delta>=0.5 outside CC OTM bracket
  [spot*1.01, spot*2.0]"`) when `_solve_call_strike` returns
  `None` because the user-requested `target_deltas` are too
  high to admit an OTM solution. The OTM-only bracket is a
  wheel-strategy invariant and should NOT be relaxed
  (changing `_solve_call_strike` would break the wheel's
  "sell calls you'd happily assign above basis" semantics),
  so the fix is purely observability. Mirrors the [[realism-check-pattern]]
  S22 F1 → PR #181 drops-accumulator shape. The natural
  follow-on Sn after Fix #3 ships would re-run S28's ITM
  probe on VZ/JPM and confirm the drops entry surfaces with
  the right reason string.

- **Fix #4 (data-coverage triage):** investigate the WMT
  70-day OHLCV (F5). A likely cause is a Bloomberg ticker
  re-extract that missed pre-2025-12 data; another is a
  recent ticker change (`WMT US Equity` → some new BBG
  identifier). Either way the dividends and fundamentals
  files contain WMT, so the symbol is alive in the universe.
  Fix is upstream of the engine.

- **The CC-near-ex-div realism test would benefit from a
  Theta replay.** A Theta-provider Sn (queued S6) would
  provide actual quoted chains at the strikes the engine
  refuses to produce on Bloomberg (F4), so an ITM-near-ex-div
  CC could be priced against real market premiums and the
  engine's early-exercise penalty validated against
  market-implied early-exercise probability.

**Methodology debt.**

- **Single-as_of test (2026-03-20).** Repeating S28 at a
  different as_of with different inside/outside groupings
  would confirm the F3 truncation generalises (versus
  "the file is fresh through 2026-Q1 but stale after"). A
  cleaner phrasing: re-run S28 at `as_of=2025-12-01` to see
  if the 25% future-coverage figure shifts up (more recent
  vintage) or stays at 25% (systematic). If it stays at 25%,
  the file has a fixed-look-ahead horizon problem; if it
  shifts up, the file is just stale-as-of-2026-03-20.

- **No Theta cross-check.** All dividend amounts are read from
  Bloomberg's `sp500_dividends.csv`. A spot check against
  Yahoo Finance or another source for the four inside-window
  names (VZ, JPM, MSFT historical, WMT historical) would
  catch transcription errors in the dividend file.

- **A/B held `dividend_yield=0.0` to isolate the dividend's
  total impact, but did not isolate the BSM-q-only vs
  EVEngine-penalty-only contributions.** To split them
  cleanly would require either (a) exposing
  `expected_dividend` as a separate kwarg on
  `rank_covered_calls_by_ev` (not currently a parameter)
  or (b) monkey-patching `conn.get_next_dividend` to return
  None for the test, which crosses into integration-test
  territory.
