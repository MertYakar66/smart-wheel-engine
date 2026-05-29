---
id: S23
title: Earnings-window navigation (event gate + IV-crush on AVGO)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Exercise the engine's earnings-aware behavior end-to-end:
event-gate boundary on `WheelRunner.rank_candidates_by_ev` across
the trading day before / day of / day after a real earnings event,
plus the IV-crush impact on the forward-distribution + strike-solve
that a wheel trader would expect to see in the ranker output. AVGO
reported 2026-03-04 (Wed) inside the data window — a clean target
for the boundary scan, with multiple post-earnings trading days
available before the 2026-03-20 OHLCV cutoff.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
Five-ticker basket: **AVGO** (target) + AAPL, MSFT, KO, HD (clean
controls with no earnings inside any tested buffered window).
Default 35-DTE / 25-delta / 5-day earnings buffer. Four `as_of`
dates probing the boundary:

- **2026-03-03** (Tue, TDB — earnings tomorrow)
- **2026-03-05** (Thu, TDA — earnings yesterday)
- **2026-03-10** (Tue, 6 calendar days post-event — first date
  outside the nominal back buffer)
- **2026-03-13** (Fri, 9 calendar days post-event — deep into the
  post-event regime)

Driver under `%TEMP%\s23\`, not committed. Pure observation —
reads `wr.rank_candidates_by_ev(...).attrs['drops']` and the
survivor frame; no `EnginePhaseReviewer` wiring, no `WheelTracker`
attached.

**Path.** `rank_candidates_by_ev` at `engine/wheel_runner.py:579`
builds a per-run `EventGate` (`engine/event_gate.py:76`) from
`conn.get_next_earnings(ticker, as_of)` per ticker
(`engine/wheel_runner.py:906`). The gate's
`_event_touches_window` (`event_gate.py:110-119`) is symmetric —
buffer applied to both `trade_start` and `trade_end`. The IV input
for the strike-solve comes from `conn.get_fundamentals(ticker)`
(`engine/wheel_runner.py:813-816`), preferring
`implied_vol_atm` then falling back to `volatility_30d`.

**Status.** Done. **Verdict: the event gate boundary fires
correctly on AVGO at the TDB (2026-03-03), then never again — and
the IV the engine uses for the strike-solve is a single snapshot
that does NOT change across the four `as_of` dates. Three
structural findings.**

**Findings:**

- **Headline cross-section.** AVGO across the four as_of dates,
  same 35-DTE / 25-delta config:

  ```
  2026-03-03 (TDB)        DROPPED  gate=event  reason=event_lockout:earnings@2026-03-04 (+/-5d buffer)
  2026-03-05 (TDA)        SURVIVED iv=0.4296   premium=$7.139   ev_dollars=$+268.85   ev_per_day=$+14.28
  2026-03-10 (6d post)    SURVIVED iv=0.4296   premium=$7.325   ev_dollars=$+310.30   ev_per_day=$+16.93
  2026-03-13 (9d post)    SURVIVED iv=0.4296   premium=$6.857   ev_dollars=$+150.46   ev_per_day= $+8.28
  ```

  The event gate fires exactly once — at the TDB — then AVGO
  reappears at the TDA and never disappears again. **Logged.**

- **(F1) `get_next_earnings` is strictly forward-only — the 5d
  back-buffer in `EventGate` is effectively dead code.**
  `engine/data_connector.py:408` filters with
  `df[df["announcement_date"] > ref]`. So at `as_of=2026-03-05`
  (the day after AVGO's 2026-03-04 earnings),
  `get_next_earnings("AVGO", "2026-03-05")` returns `None` — the
  just-passed earnings event is never registered on the
  `EventGate`, so the symmetric back-buffer at
  `engine/event_gate.py:117` never has any past event to test
  against. **The 5d back buffer is unreachable in production.**
  Live probe from the driver:

  ```
  as_of=2026-03-03  next_earnings_after={'announcement_date': Timestamp('2026-03-04 ...')}
  as_of=2026-03-04  next_earnings_after=None
  as_of=2026-03-05  next_earnings_after=None
  as_of=2026-03-10  next_earnings_after=None
  as_of=2026-03-13  next_earnings_after=None
  ```

  **Either** the back-buffer was intended to fire on just-passed
  earnings (and the `>` filter at `data_connector.py:408` is a bug
  — it should be `>=` minus the buffer, or the gate should pull
  past events too), **or** the back-buffer is intentionally
  dormant (the wheel trader can write into post-earnings IV crush
  opportunistically). The current code says one thing
  (symmetric buffer) and the connector says another (forward-only
  feed) — they disagree. **Logged as a structural inconsistency.**

- **(F2) Bloomberg earnings CSV is forward-truncated for many
  tickers — silent event-gate bypass.** Live driver probe at
  `as_of=2026-03-20`:

  ```
  AAPL  : next after 2026-03-20 = None
  MSFT  : next after 2026-03-20 = None
  GOOGL : next after 2026-03-20 = None
  AVGO  : next after 2026-03-20 = None
  COST  : next after 2026-03-20 = None
  ORCL  : next after 2026-03-20 = None
  ```

  AAPL/MSFT/GOOGL/AVGO/COST/ORCL all have well-known late-April
  2026 earnings in real life, but the Bloomberg earnings CSV in
  the repo (`data/bloomberg/sp500_earnings.csv`, last row
  2026-03-31) has no entries past mid-March for these names. The
  consequence: at `as_of=2026-03-20`, the event gate is a **no-op**
  on six of the top-ten S&P 500 names. A trader running a 35-DTE
  ranker at the data cutoff would freely open AAPL/MSFT/GOOGL
  positions whose holding window crosses their real earnings. **The
  event gate's silent-on-no-data behavior makes this invisible**:
  no drop entry, no warning, just no event registered. By
  contrast, XOM (2026-04-07 earnings IS in the CSV), JPM
  (2026-04-14), UNH (2026-04-21), and JNJ (2026-04-14) DO get
  blocked correctly — the gate is doing its job when the data is
  there. **Logged as a data-completeness vs. observability gap.**

- **(F3) The IV input to the strike-solve is NOT PIT-aware.**
  AVGO surfaced with `iv=0.4296` (= 42.96%) at **all four** of
  the post-event `as_of` dates — even though the connector's
  `get_iv_history` shows the put-IV moving meaningfully:

  ```
  date        hist_put_imp_vol  volatility_30d
  2026-03-03           55.935          36.208
  2026-03-04           52.957          36.309    (earnings day)
  2026-03-05           49.547          38.926
  2026-03-10           48.437          40.184
  2026-03-13           49.819          42.414
  2026-03-20           46.503          35.010
  ```

  The 42.96% value matches `conn.get_fundamentals("AVGO")
  ['implied_vol_atm']` exactly — and that connector method
  (`engine/data_connector.py:590`) takes **no `as_of` argument**
  and reads from a snapshot fundamentals CSV
  (`sp500_fundamentals.csv`) that has **no date column** at all:

  ```
  >>> conn.get_fundamentals('AVGO')
  {..., 'volatility_30d': 34.875, 'implied_vol_atm': 42.9566, ...}
  ```

  So the engine's per-call IV is frozen — same value at
  `as_of=2026-02-13` as at `as_of=2026-03-20`. **The "PIT-safe"
  claim in `rank_candidates_by_ev` (line 607 of the docstring)
  is true for OHLCV and the empirical forward distribution
  derived from it, but the IV used for the strike-solve and the
  BSM-fair premium is a snapshot.** The IV-crush experiment is
  literally not observable through the ranker output.

  This isn't theoretical — it shapes the result. At 2026-03-05
  AVGO's *real* IV was 49.5% (immediately post-crush spike); at
  2026-03-13 it was 49.8%; at 2026-03-20 it was 46.5%. The
  engine used 42.96% throughout — too low at the TDA, roughly
  right at 2026-03-20. The strike solved at the wrong IV (lower
  than reality at the TDA) is too far OTM, the synthetic premium
  is undershoot, and `ev_dollars` is mispriced versus what the
  trader would actually transact at. **Logged as the highest-
  leverage finding in S23.**

- **The `WheelTracker._connector_atm_iv` helper at
  `engine/wheel_tracker.py:1344` already does the right thing for
  mark-to-market** — it pulls
  `conn.get_iv_history(ticker, end_date=as_of)` and takes the
  most recent row, normalising percent→decimal. The same helper
  pattern would solve F3 inside `rank_candidates_by_ev`. Cross-
  reference: `rank_covered_calls_by_ev` (used in S22) and
  `rank_strangles_by_ev` likely share the same IV-snapshot bug
  (both use the same `conn.get_fundamentals` fallback per
  `engine/wheel_runner.py:1913` and `:2377`), not exercised
  separately in S23. **Logged.**

- **§2 verified.** Every candidate that surfaced as tradeable
  (positive `ev_dollars`, no event drop) routed through
  `EVEngine.evaluate` — per the engine's standard ranker contract.
  The findings above are about **what IV the engine evaluated
  WITH**, not about a bypass of `evaluate`. No §2 violation.
  **Logged as a positive.**

- **Control names behave correctly.** AAPL/MSFT/KO/HD survive at
  all four `as_of` dates with no event-gate drops — none of their
  earnings fall inside the buffered window for any of the four
  tested dates (AAPL/MSFT Jan 2026, KO 2026-02-10, HD
  2026-02-24, all sufficiently before; their next April-2026
  earnings are not in the CSV, which is finding F2). **Logged.**

- **`±` cp1252 mangle on `event_lockout` reason strings**
  re-confirmed in this run — `event_lockout:earnings@2026-03-04
  (±5d buffer)` rendered as `... (�5d buffer)` on the Windows
  console. Producer-side one-character fix. **Logged.**

**Verdict.**

- **Event-gate FORWARD buffer behaves correctly.** AVGO blocked
  at TDB (2026-03-03) with the expected `event_lockout` reason.
  Standard wheel-trader expectation met.

- **Event-gate BACK buffer is dead code.** `get_next_earnings` is
  strictly forward-only, so the symmetric 5d-back logic at
  `event_gate.py:117` has nothing to trigger on. Either fix the
  connector or document the asymmetry — currently the code reads
  symmetric and the behavior is forward-only. **Structural
  inconsistency.**

- **The earnings CSV is incomplete for major tickers' April-2026
  reports** — silent event-gate bypass on AAPL/MSFT/GOOGL-class
  names at the data cutoff. Data refresh, not engine, but
  surfaces a brittle "silent-on-no-data" contract. **Logged.**

- **The IV input to the put-entry strike-solve is a single
  snapshot, NOT a PIT-aware time-series.** The engine ranker
  cannot reflect IV-crush at all; the value is frozen between
  fundamentals refreshes. Material to anyone trading around
  earnings. **Logged as the highest-leverage finding.**

**AI handoff.**

- **F3 (IV snapshot) fix sketch — promote `rank_candidates_by_ev`
  to use the same PIT-aware IV helper `WheelTracker` already
  uses.** Today, at `engine/wheel_runner.py:813-816`:

  ```python
  fundamentals = conn.get_fundamentals(ticker) or {}
  iv_raw = fundamentals.get("implied_vol_atm")
  if iv_raw is None or (isinstance(iv_raw, float) and np.isnan(iv_raw)):
      iv_raw = fundamentals.get("volatility_30d")
  ```

  Proposed (mirrors `WheelTracker._connector_atm_iv` at
  `engine/wheel_tracker.py:1344-1383`):

  ```python
  iv = None
  if hasattr(conn, "get_iv_history"):
      try:
          hist = conn.get_iv_history(ticker, end_date=as_of)
          if hist is not None and not hist.empty:
              cols = [c for c in ("hist_put_imp_vol", "hist_call_imp_vol")
                      if c in hist.columns]
              if cols:
                  row = hist.iloc[-1]
                  vals = [float(row[c]) for c in cols if pd.notna(row[c])]
                  if vals:
                      iv = sum(vals) / len(vals)
      except Exception:
          iv = None
  if iv is None:
      fundamentals = conn.get_fundamentals(ticker) or {}
      iv_raw = fundamentals.get("implied_vol_atm")
      if iv_raw is None or (isinstance(iv_raw, float) and np.isnan(iv_raw)):
          iv_raw = fundamentals.get("volatility_30d")
      iv = float(iv_raw) if iv_raw is not None else 0.0
  # Existing percent->decimal normalisation continues below.
  ```

  Mirror change in `rank_covered_calls_by_ev`
  (`engine/wheel_runner.py:1913`) and `rank_strangles_by_ev`
  (`engine/wheel_runner.py:2377`). **Decision-layer surface — needs
  a decision-layer lock claim and regression coverage that the IV
  used by the ranker matches `get_iv_history(ticker, as_of).iloc[-1]`
  when both are available.** Out of scope for this Sn.

- **F1 (back-buffer dead code) options.** Either (a) fix
  `get_next_earnings` to return events within `as_of - max_buffer`
  through the future, so the back-buffer in the gate has events
  to test, or (b) document the asymmetry and remove the
  back-buffer arithmetic to avoid the misleading code. (a) is
  the trader-intent-preserving fix (block writes immediately
  post-earnings until the news / IV-crush absorbs); (b) is the
  honest-code fix if "write into the crush" is the actual
  policy. **Design call**, not a usage-test fix.

- **F2 (forward-truncated earnings CSV) is a data refresh.** The
  Bloomberg earnings file ends mid-March 2026 for AAPL-class
  tickers; the next April earnings need to be pulled. **Data,
  not engine.** Tracked under the existing Bloomberg-refresh
  memory.

- **A regression test that would have caught F3** — a unit test
  asserting that, for a fixed ticker and two `as_of` dates with
  different `hist_put_imp_vol` values in `sp500_vol_iv_full.csv`,
  the `iv` column in `rank_candidates_by_ev`'s output differs.
  Today's behavior would fail that assertion. Out of scope for
  this Sn.

**Methodology debt.**

- **Single ticker, single earnings event.** S23 ran AVGO only.
  COST (2026-03-05), ORCL (2026-03-10), LULU (2026-03-17),
  MU (2026-03-18) are all in the data window and would let the
  finding be replicated across more events. **Logged.**

- **The `regime_multiplier` column was empty in the survivor
  output** (omitted from the printed columns because the
  diagnostic column wasn't populated by the engine on this
  basket). Whether that's an HMM-cold-start issue (no persisted
  model in `models/`) or by design at this basket size isn't
  exercised here. **Logged for a future Sn** — overlaps with the
  ruled-out scenario E (steady-state regime sizing).

- **R7 / R8 not exercised in S23** — no `PortfolioContext`
  attached, no `EnginePhaseReviewer` wired. S21 covered them at
  a different angle; S24 (multi-strategy book) will exercise
  them on a richer book.

- **Ruled out per the campaign constraints:** Theta provider,
  decision-layer code change (S23 found gaps, did not fix), the
  HMM regime path (no persisted model), the dashboard surface.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — every AVGO survivor at the post-event dates routed through `EVEngine.evaluate` (PIT-IV input now).
  - Qualitative verdict: **partial — both major findings F1 and F3 are CLOSED**:
    - **F1 (event-gate BACK buffer dead code) — CLOSED**: `MarketDataConnector.get_recent_earnings(...)` now exists on `main` (Terminal B's symmetric-gate work, PR #180). The 5-day back-buffer is now reachable. In this re-run, `as_of=2026-03-05` (1 calendar day after AVGO's 2026-03-04 earnings) yields **0 ranked rows with an event-gate drop** — exactly the symmetric back-buffer the original entry said was dead. (Originally `as_of=2026-03-05` surfaced AVGO with `ev_dollars=+268.85`; post-#180 it's blocked).
    - **F3 (IV input not PIT-aware) — CLOSED**: AVGO's `iv` column at the two post-event dates **now differs**: `iv=0.4844` at 2026-03-10, `iv=0.4982` at 2026-03-13. Originally both used `iv=0.4296` (the snapshot). The current values agree with the IV history file's `(hist_put_imp_vol + hist_call_imp_vol) / 2` per PR #179.
  - Numerical drift > 5% (with attribution):
    - metric `AVGO_iv[2026-03-10]`: orig `0.4296` → new `0.4844` (`+12.8%`); attributable to **PR #179** (`_resolve_pit_atm_iv`).
    - metric `AVGO_iv[2026-03-13]`: orig `0.4296` → new `0.4982` (`+16.0%`); attributable to **PR #179**.
    - metric `AVGO_ev_dollars[2026-03-10]`: orig `+310.30` → new `+390.06` (`+25.7%`); higher PIT-IV raises the synthetic premium and EV.
    - metric `AVGO_ev_dollars[2026-03-13]`: orig `+150.46` → new `+222.72` (`+48.0%`); same direction.
    - metric `AVGO_at_2026-03-05_survived`: orig **True** (iv=0.4296, ev=+268.85) → new **False** (dropped on event_lockout back-buffer); attributable to **PR #180** (symmetric event gate via `get_recent_earnings`).
  - Notes: F2 (Bloomberg earnings CSV forward-truncated past mid-March 2026) is a data-completeness issue, not engine — same state on the worktree's CSV.
