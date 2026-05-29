---
id: S18
title: Load / scale stress (production-scale SP500 characterisation)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Characterise the engine under heavy load — full-universe
(503-ticker SP500) rank calls, repeated calls in succession, deep
dossier batches, deep histories, wide `top_n` — and document latency,
peak memory, file-handle counts, intra-process cache growth between
calls, and whether the engine fails fast or degrades silently when
pushed past its comfort zone. Pro question: would this survive a real
production deployment running across the full SP500 universe at scale,
or does something break, leak, or drift?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`, full SP500 universe via
`runner.connector.get_universe()` → **503 tickers**. Five load lanes
(L1–L5 + L5b overshoot probe), one `WheelRunner` instance across L1–L5
to measure intra-process cache growth and warm-cache convergence.
Throwaway instrumentation script (not committed; same pattern as
S5/S15/S16/S17). **Platform: Windows (CI runs on Linux);**
`resource.getrusage` and `/proc/<pid>/fd` are Unix-only, so a `ctypes`
PSAPI wrapper handles system RSS (`GetProcessMemoryInfo.WorkingSetSize`
and `.PeakWorkingSetSize`) and open kernel handles
(`GetProcessHandleCount`). `tracemalloc` gives the cross-platform
Python-heap-peak. Engine-side numbers are platform-equivalent; OS-side
absolute values will differ slightly from a Linux CI run by a small
constant.

**Path.** Each lane wraps `runner.rank_candidates_by_ev(...)` or
`build_dossiers(...)` in matched `snap()` / `diff()` calls. `snap()`
captures wall-clock (`time.perf_counter`), `WorkingSetSize` /
`PeakWorkingSetSize` (MB), `tracemalloc.get_traced_memory()` peak (MB),
open handle count, `gc.get_count()`, and `len(gc.get_objects())`. The
WheelRunner instance is introspected for any attribute that is a
`dict`/`list`/`set` and either starts with `_` or has "cache" in the
name — only `_hmm_regime_cache` (`engine/wheel_runner.py:254`) qualifies
on a non-Theta runner. Warnings captured via a `warnings.showwarning`
override.

**Status.** Done. **Verdict: YES — would survive production-scale SP500
deployment as-is, with three named conditions.**

1. **Warm the HMM regime cache once at startup.** Cold-call latency is
   145 s for a 503-ticker full-universe rank; subsequent calls on the
   same `WheelRunner` instance converge to **~10.5 s steady-state**
   (14× speed-up). Production should run a warm-up rank at boot, not
   on the first user request.
2. **Mild handle leak (~5 handles per repeated call).** Not exponential,
   not file-descriptor-exhaustion-level, but real: L2 calls each added
   +5 handles. At 100 calls/day this is ~500 handles/day on a process
   with a Windows default 16,384 handle cap. Investigate before
   high-throughput deployment; until then, plan for a daily process
   restart.
3. **Budget ~800 MB peak RSS during startup,** ~450 MB steady-state.
   The high-water mark hits during initialization (HMM cache load),
   not during steady-state ranking. A 1 GB working-set budget is
   comfortable.

§2 verified — no observed code path emits a tradeable verdict without
upstream `EVEngine.evaluate`; no §2 bug surfaced during the read.

**Findings:**

- **Per-lane operational table** (Bloomberg `as_of=2026-03-20`, 503-ticker
  universe, all lanes one process / one `WheelRunner`).

  | Lane | Description | Wall | RSS Δ | Handles Δ | Survivors | Notes |
  |---|---|---|---|---|---|---|
  | init | `WheelRunner()` + `get_universe()` | 13.0 s | +243 MB | +0 | – | Memory dominant cost is here; peak RSS hits 805 MB during this step (tracemalloc Python-heap peak +315 MB) |
  | **L1** | full-universe rank `top_n=50` cold | **145.2 s** | +13.2 MB | +18 | 50 / 70 drops | `_hmm_regime_cache` 0 → **492** (one entry per non-dropped ticker); gc objects +10,242 |
  | **L2** | same runner, repeat L1 ×5 | **10.5–10.7 s** (each) | +0.0 MB | +5 (each) | 50 / 50 / 50 / 50 / 50 | cache stays at **492 across all 5**, no growth; per-call latency 10.337 / 10.584 / 10.521 / 10.548 / 10.337 — extremely stable |
  | **L3** | `build_dossiers` on L1 top 50 | **0.059 s** | +0.0 MB | +0 | 50 verdicts | All 50 → `review` / `chart_context_missing` (offline `FilesystemChartProvider` — expected R2 per S16) |
  | **L4** | deep-history rank, 5 megacaps × 2065 days | **0.49 s** | +0.0 MB | +5 | 4 / 1 drop | Cache hits on the 5 names; sub-100 ms per ticker on full 2065-bar history. No O(n²) hotspot observed |
  | **L5** | full-universe rank `top_n=500` warm | **10.5 s** | +1.4 MB | +5 | 433 / 70 | Same wall-clock as L2 — `top_n` does not materially affect cost; the rank computation is the cost, row emission is free |
  | **L5b** | overshoot probe `top_n=10_000` | 10.8 s | – | – | caps to 433 | Graceful — no exception, no warning, ranker correctly caps to actual survivor count |

  Final state after all lanes: RSS 441.6 MB, peak RSS 805.6 MB,
  handles 400, gc_objects 117,786. Total wall 234.9 s.

- **Cold vs warm latency profile — 14× speed-up.** L1 cold = 145.2 s;
  L2 warm = 10.5 s. The cold-call cost is dominated by the HMM 4-state
  Gaussian model fit for each non-dropped ticker (`engine/regime_hmm.py`
  via `wheel_runner.py:1040–1060`). Once each `(ticker, history_days)`
  key is cached, the next call is essentially the BSM cost +
  forward-distribution per candidate. A `(ticker, history_days)` cache
  key means a *fresh history window* (i.e., a new trading day's worth
  of data) misses the cache and re-fits — daily production cadence
  will pay the cold-load cost **once per day**, not once per request.
  **Logged.**

- **`_hmm_regime_cache` is bounded.** Converges to **492 entries** on
  the first full-universe call (one per ticker that survives the
  ohlcv/history precondition) and stays there across L2's 5 repeated
  calls. No unbounded growth. The cache key is `(ticker,
  history_days)` so the entry count is bounded by `|universe| ×
  |distinct history window sizes|` — under realistic daily-batch
  operation, that's just `|universe|` per day. **Logged.**

- **`peak_RSS_delta=0` on every lane after init.** Peak RSS was set
  during `WheelRunner()` construction (805.6 MB) and never exceeded by
  any lane. Steady-state RSS settles at ~441 MB. The init memory cost
  comes from loading Bloomberg CSVs into the connector
  (`sp500_ohlcv.csv` 59 MB + `sp500_vol_iv_full.csv` 78 MB + a
  fundamentals/macro frame). `tracemalloc` confirms +315 MB Python-heap
  peak during init, congruent with the ~243 MB system-RSS delta (the
  difference is unfreed pandas internals). **Logged.**

- **Mild handle leak — +5 handles per L2 call, +18 on cold L1.** Not
  exponential and not file-descriptor exhaustion-level, but
  measurable. Over the 7 ranker calls in this run (L1 + L2×5 + L5 +
  L5b) the handle count walked from baseline post-init to 400. The
  per-call +5 strongly suggests a fixed number of file or kernel
  handles opened per call that aren't always closed — most likely the
  IV-history / OHLCV CSV reads materialise `pd.read_csv` handles that
  pandas closes lazily, or the `_hmm_regime_cache` keeps an
  `HMMRegimeDetector` instance alive that holds references. Not an
  emergency at the observed rate (~500 handles/day at 100 calls), but
  worth a follow-up read of the connector / HMM code paths.
  **Logged.**

- **`top_n` is not the cost.** L5 (`top_n=500` warm) = 10.5 s = L2
  (`top_n=50` warm). The rank computation runs over the full universe
  regardless of `top_n`; the sort+head at the end is essentially free
  on a 433-row frame. Implication: a pro running the ranker once and
  consuming the top-50 has paid the same compute cost as a pro asking
  for the top-500. **Logged.**

- **`top_n=10_000` overshoot is graceful.** No exception, no warning;
  the ranker returns 433 rows (the actual survivor count). Production
  callers asking for "all survivors" by passing a large `top_n` get
  exactly that, capped to the universe. **Logged as a positive.**

- **Deep history does not blow up.** L4 ranked 5 megacaps with 2065
  trading days of history each (the full Bloomberg coverage) in
  0.49 s total — about 100 ms per ticker. With the HMM cache warm,
  the per-ticker work is BSM + forward distribution on the full
  history, and that's linear in days, not quadratic. The HAR-RV /
  block-bootstrap / POT-GPD path on 2065 bars is well within budget.
  **Logged as a positive.**

- **70 drops on L1 = ~14% of universe** (`as_of=2026-03-20` is dense
  in Q1 earnings lockouts per S16; same shape here). Every drop has
  a `{ticker, gate, reason}` record in `.attrs["drops"]` —
  **433 survivors + 70 drops = 503**, fully accounted for. No silent
  drops observed on the full-universe path. (S16's structured-drops
  finding still applies: `reason` is free text.) **Logged as a
  positive.**

- **Zero captured warnings across 234 s of load.** The
  `warnings.showwarning` override saw no DeprecationWarning,
  RuntimeWarning, FutureWarning, or otherwise. The 5-ticker smoke
  warning surface (S17 also reported zero) extends to the
  503-ticker full-universe surface. **Logged as a positive.**

- **Init is the memory peak; per-call deltas are de minimis.** Cold
  init adds +243 MB RSS / +315 MB Python heap; steady-state per-call
  RSS Δ is 0.0 MB on all 5 L2 repeats. Memory is well-bounded under
  load. **Logged.**

- **No new orphaned surface noticed at scale.** The S15-style "exists
  but unwired" pattern (RiskManager / SectorExposureManager) was the
  only one observed across the campaign; this run didn't surface a
  new sibling. The decision-layer is fully wired through
  `WheelRunner.rank_candidates_by_ev` → `EVEngine.evaluate` from the
  full-universe entry point. **Logged.**

- **§2 verified across the load.** Every survivor on every lane came
  through `rank_candidates_by_ev` → `EVEngine.evaluate`; the
  reviewer-applied dossier verdicts on L3's 50 candidates were all
  R2 `chart_context_missing` (downgrade-only, expected). No path
  emitted a tradeable outcome without upstream EV. **No §2 bug
  surfaced; no regression test added.**

**AI handoff.**

- **Handle-leak follow-up.** The +5 handles per call (and +18 on the
  cold call) is the one mild concern. Suggested probe: wrap a single
  `rank_candidates_by_ev` call in `tracemalloc.get_traced_memory()`
  + a `gc.get_referrers` snapshot to identify which objects accumulate
  per call. A likely suspect is `engine/regime_hmm.py`'s detector
  cache holding open references to history `DataFrame`s. Out of scope
  for S18 — flagged for a small read-only follow-up Sn or a Terminal-
  A-lane decision-layer touch if the root cause is in `wheel_runner.py`.
- **Production warm-up pattern.** A `WheelRunner.warm()` method that
  loads the universe and populates `_hmm_regime_cache` synchronously
  at process start would hide the 145 s cold cost from user-facing
  request latency. Not claimed; flagged as the natural fix for the
  cold-load condition above.
- **L5b's `top_n=10_000` graceful-cap behaviour** is a structural
  positive worth pinning with a small regression test. Out of scope
  for a read-only usage test; flagged for a future Terminal A test
  addition.
- **Theta provider scale-out** is out of reach in this Cowork sandbox.
  The Bloomberg path has been characterised end-to-end here; the
  Theta path would have a fundamentally different latency / memory
  profile (live chains, persistent HTTP session to `:25503`,
  per-strike fetches) and warrants a separate Sn on the laptop with
  Theta Terminal up. **Theta-blocked.**

- **Ruled out per the prompt (don't litigate):** decision-layer edits,
  cProfile / line_profiler / memray (stdlib was sufficient), network
  load against `engine_api.py :8787` (S20's lane), failure-mode chaos
  / malformed payloads / corrupted-spot injection (S19's lane), Theta
  provider (operator-gated), optimisation / refactor.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — full-universe rank surfaces nothing tradeable without `EVEngine.evaluate`; structured drops on every dropped ticker.
  - Qualitative verdict: **partial — verified live, but the warm/cold latency profile has shifted materially**. L1 cold: 79.3 s (was 145.2 s, **45% faster**). L2 warm: 41.2 s (was **10.5 s, 4× SLOWER**). HMM cache 491 (was 492 — within 1 entry of the documented 492). RSS post-init: 672 MB (was 805 MB peak / 441 MB steady-state — current 672 is partway between, single-run snapshot). `top_n=10_000` overshoot still caps gracefully to actual survivors (423 vs original 433 — same family). Drops on the universe rank are still all-or-nothing structured.
  - Numerical drift > 5% (with attribution):
    - metric `L1_cold_wall_s`: orig `145.2` → new `79.3` (`-45.4%`); attributable to a **composite** of HMM caching improvements between S18 and now plus L1 hardware-state variance — not pinning to a single PR. The HMM cache size delta (492 → 491) is within seed-stable refit noise.
    - metric `L2_warm_wall_s`: orig `10.5` → new `41.2` (`+292%`); attributable to **PR #215 + PR #220** (`as_of-beyond-data` refusal guards added per-call cost — staleness probe on every ticker + post-PR-#208 / #210 / #222 emitted diagnostic columns adding serialization overhead). This is a **real warm-path regression** worth surfacing — the docstring-documented "warm calls ~10s" is no longer accurate. Flagged for follow-up; not in scope for this re-verify.
    - metric `L5b_top_n_10000_survivors`: orig `433` → new `423` (`-2.3%`); within Bloomberg-data-window noise / earnings-calendar drift on `2026-03-20`.
  - Notes: handle-leak follow-up (+5 handles per call documented in original entry) not measured this pass — would require a 100+-call sweep; the warm-path regression is the more pressing finding.

---
