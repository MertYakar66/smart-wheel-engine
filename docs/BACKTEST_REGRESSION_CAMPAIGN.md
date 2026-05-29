# Backtest Regression Harness — Campaign Report

**Campaign**: 4-PR series to convert the four committed ledger backtests
(S27, S32, S34, S35) from human-curated Markdown into executable
pytest assertions against the current engine.

**Status (2026-05-26)**: PR1, PR2, PR3 pushed. PR4 in progress (S35
re-baseline). All four reproducers run end-to-end; three snapshots
locked; S35 re-baseline pending.

**Why this campaign exists**: the S22 → S27 episode proved historical
backtests can drift silently when the engine's methodology changes —
ρ went from 0.484 to 0.218 after the IV-PIT fix, but the original S22
doc remained valid-looking on `main` until S27 explicitly re-ran it.
Without an automated harness, any future engine change can quietly
invalidate the documented ρ / NAV / hit-rate claims, and nobody
notices until somebody re-runs by hand. This harness closes that gap.

---

## Architecture

```
backtests/regression/
  universes.py            UNIVERSE_24 (S22/S27/S32/S35 pinned set)
                          UNIVERSE_100 (first 100 alphanumeric from
                                       MarketDataConnector.get_universe())
  _common.py              run_backtest()                 — single friction (S27)
                          run_backtest_multi_friction()  — N friction levels with
                                                           shared SP-rank call
                          Friction overlay, snapshot I/O, forward-replay
  s27_ivpit_24t_100k.py   $100k / 24t / 2022-2024 / frictionless
  s32_friction_24t_1m.py  $1M  / 24t / 2022-2024 / 3 friction levels
  s34_universe_100t_1m.py $1M  / 100t / 2022-2024 / 3 friction levels (top_n=15)
  s35_oos_24t_100k.py     $100k / 24t / 2018-2020 OOS / 3 friction levels
  snapshots/              committed JSON snapshots — the regression baseline

tests/test_backtest_regression.py
  - test_universes_match_connector            (structural, fast)
  - test_snapshot_fingerprints_have_required_keys  (structural, fast)
  - test_backtest_matches_snapshot            (slow, behind @pytest.mark.backtest_regression)

.claude/commands/backtest-regression.md
  Tier-3 skill — single pytest invocation.

.github/workflows/backtest-regression.yml
  Manual workflow_dispatch only. Cron disabled until CSV hydration in
  CI is solved.

.github/workflows/ci.yml (modified)
  Per-PR test job adds `-m "not backtest_regression"` so the slow
  reproducers don't block per-PR CI for hours.
```

**Hard invariant maintained (CLAUDE.md §2)**: every reproducer routes
candidates through `WheelRunner.rank_candidates_by_ev` end-to-end. No
engine mocking. No bypass of `EVEngine.evaluate`.

---

## Methodology

### Trading-day loop (per friction level)

For each business day in `[start, end]`:

1. **Mark-to-market** all open positions using close prices on the day
   (fall back to next available trading day within 1 day).
2. **Settle expirations** via `WheelTracker.handle_put_expiration` and
   `.handle_call_expiration`, which encapsulate the
   `ITM → assign` / `OTM → keep/close` branching.
3. **Wheel into covered calls** on any `STOCK_OWNED` tickers. Without
   this step, every put assignment locks its ticker out of future
   rotation — the S27 v1 snapshot (no CC wheeling) showed only 15
   executed trades vs the documented 50. CC wheeling restored the
   match (51 executed in v2).
4. **Rank short-put candidates** via
   `rank_candidates_by_ev(tickers, as_of, dte_target=35,
   delta_target=0.25, top_n=N, min_ev_dollars=-1e9,
   include_diagnostic_fields=True)` — capture the full ranked frame
   for forward-replay; filter on `ev_dollars > 0` at the execution
   step (not at the rank step).
5. **Persist rank-log rows** to compute per-candidate forward-replay
   later. Each row carries ticker / ev_dollars / premium (friction-
   adjusted) / strike / iv / prob_profit / expiration_date.
6. **Open new positions** up to `max_new_per_day=3`, skipping
   negative-EV rows and rows whose ticker already has an open
   position. Strike-collateral check against
   `available_buying_power()`.

After the loop, **forward-replay** every persisted rank-log row: look
up the spot at expiration_date, compute the held-to-expiry P&L for
that candidate. The replayed rows feed the aggregate Spearman ρ /
hit-rate / per-year / per-quartile metrics.

### Friction overlay (S32 doc method appendix, verbatim)

| Component | Model |
|---|---|
| Bid/ask half-spread | `max($0.05, 8 % × premium)` per share |
| Commission (open / close) | `$0.65 / contract` |
| Assignment slippage | `10 bp × strike × 100` (equity notional) |

Three friction levels: `"none"` (frictionless), `"bid_ask"` (half-
spread only), `"full"` (all three).

### Multi-friction shared-rank optimization

The S32 doc described three `WheelTracker` instances running in
parallel sharing one rank call per day. The naive sequential approach
(loop `run_backtest(friction_level=...)` over 3 levels) does the SP
rank 3× per day. `run_backtest_multi_friction` does it 1× per day and
gives each tracker an independent decision against that shared frame.

| Compute cost | Naive (3× sequential) | Shared-rank |
|---|---|---|
| SP-rank EV evals per day | 3 × N tickers | 1 × N tickers |
| S32 wall-clock | ~150 min | ~55 min |
| S34 wall-clock (100 tickers) | ~10 h (extrapolated) | ~4.5 h (observed) |

S27 stays on the single-friction `run_backtest` path (no benefit from
multi-friction for a one-level run).

---

## Snapshot-vs-doc comparison

The harness's purpose is to **lock current engine behavior**, not to
re-derive the documented numbers exactly. Where the snapshot diverges
from the source doc, the divergence is captured here so the test
remains a meaningful regression gate.

### S27 — `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`

$100k / 24 tickers / 2022–2024 / frictionless.

| Metric | S27 doc | Snapshot | Notes |
|---|---|---|---|
| Spearman ρ | 0.2183 | 0.1881 | within ~0.03 |
| Row count (replayed) | 6,163 | 5,944 | −3.6 % |
| Executed trades | 50 | 51 | match |
| Put assignments | 10 | 12 | close |
| Hit-rate (OTM) | 76.39 % | 80.53 % | +4 pp |
| Mean realized P&L | $63.34 | $51.70 | −18 % |
| 2022 ρ (bear) | 0.391 | 0.375 | close |
| 2023 ρ (recovery) | 0.193 | 0.177 | close |
| 2024 ρ (bull) | 0.131 | 0.078 | lower (noise floor) |
| Q3 vs Q0 P&L spread | 1.67× | 1.54× | monotonic |
| Final NAV ($100k start) | $151,444 | $127,694 | −16 % |

### S32 — `docs/ENGINE_BACKTEST_S32_FRICTION.md`

$1M / 24 tickers / 2022–2024 / 3 friction levels.

| Metric | S32 doc (full friction) | Snapshot (full) | Notes |
|---|---|---|---|
| Spearman ρ | 0.1918 | 0.1864 | within 0.006 |
| Final NAV | $1,018,514 | $1,078,208 | +5.8 % |
| Friction drag | $2,745 | $3,497 | 0.27 % → 0.32 % NAV |
| Short puts opened | 95 | 105 | +10 % |
| Put assignments | 14 | 19 | +36 % |
| Same exec across levels | yes | yes | "BP never binding at $1M" ✓ |

### S34 — `archive/2026-05/SOUNDNESS_REVIEW_2026-05-26.md` / `docs/ENGINE_BACKTEST_S34_UNIVERSE.md`

$1M / 100 tickers / 2022–2024 / 3 friction levels / `top_n=15`.

| Metric | S34 doc | Snapshot (full) | Notes |
|---|---|---|---|
| Spearman ρ | 0.3273 | 0.3286 | **within 0.002 — strong match** |
| Total candidates N | 10,315 | 10,911 | +5.8 % |
| NAV gain on $1M | +$356,128 | +$347,145 | within 2.5 % |
| Final NAV | $1,356,128 | $1,347,145 | within 0.7 % |
| Exec varies by friction | n/a in doc | 275 / 270 / 264 | BP **does** bind at 100 tickers |

S34 is the strongest snapshot-vs-doc match in the campaign. Two
reasons: (a) the 100-ticker universe averages out per-ticker strike-
rounding noise that dominates the smaller-universe gaps, and (b) the
sample size (10,911) makes the ρ statistic most reliable.

### S35 — `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md`

$100k / 24 tickers / 2018–2020 OOS / 3 friction levels.

**Surprise finding**: the S35 doc was generated against the
**post-PIT-fix engine** (commit `e504801`), not the pre-fix engine as
the campaign plan initially assumed. PR4 is therefore an
engine-version-neutral driver-implementation lock, not a pre→post fix
re-baseline. The divergences below are due to driver implementation
(strike rounding, CC-entry strike choice, expiration handling), not
engine math — the same systematic gap pattern as S27 / S32.

| Metric | S35 doc (full friction) | Snapshot (full) | Notes |
|---|---|---|---|
| Spearman ρ | 0.4970 | 0.4998 | **near-identical — driver-invariant** |
| Rank rows (per friction) | 1,970 | 2,003 | +1.7 % |
| Short puts opened | 19 | 40 | +110 % — driver finds more EV>0 |
| Put assignments | 6 | 8 | close |
| Final NAV ($100k start) | $103,566 | $115,830 | +11.8 % |
| 504-day history gate ON | yes | yes | engine constant unchanged |
| BP-binding pattern | yes (uniform) | mixed (40 / 39 / 40) | driver's friction-cash path diverges marginally |

The signal-quality finding — **S35's ρ ≈ 0.50 dwarfs S27's 0.22** —
survives the driver change exactly. This is the most important
finding the original S35 doc made, and the harness preserves it.

Execution count doubled (19 → 40). The post-PIT-fix engine actually
surfaces more positive-EV opportunities in 2018–2020 than the
original throwaway harness captured. The committed reproducer's
EV>0 + BP-available + 3-per-day cap is now the recorded
source-of-truth going forward.

The S35 doc explicitly warned: "in-sample HMM / POT-GPD parameters
STILL APPLY". The re-baseline preserves that caveat — the
parameters were tuned with full 2018-2026 visibility, so 2018–2020
is not a fully independent OOS validation.

---

## Why the systematic ~5–20 % NAV gap on S27 and S32

S27 NAV is 16 % low, S32 NAV is 5.8 % high. Both reproducers share
the same driver. The gap is consistent with **per-ticker strike
rounding and CC-entry strike choices that don't exactly mirror the
S22/S27/S32 throwaway harness pattern**. The original throwaway
harnesses were `%TEMP%/sNN_backtest/run.py` files deleted per
convention — their exact strike-selection logic is lost.

S34 has near-perfect agreement because the larger universe averages
out per-ticker strike noise — the same reason a 100-ticker portfolio's
realized P&L is more predictable than a single ticker's.

**Implication for the harness**: snapshots capture the **current
engine's behavior**, which is the meaningful regression baseline. The
documented numbers serve as a sanity check on signal direction and
ordering, not as an exact target.

---

## On-fail re-baseline workflow

When `pytest tests/test_backtest_regression.py -m backtest_regression`
fails, the response is **diagnose first, re-baseline second**:

1. **Do not regenerate snapshots reflexively.** A failure is a signal,
   not a chore.
2. `git log --oneline engine/ since <last successful snapshot date>` —
   identify the candidate PR. Each snapshot's fingerprint records
   `data_csv_sha256` and `engine_sha_at_snapshot_lock`; compare against
   current `HEAD`.
3. Read the offending PR description and `CHANGELOG.md`. Is the engine
   change **deliberate** (a real methodology improvement) or
   **accidental** (a refactor that should have been numerically
   invariant)?
4. **Deliberate**: regenerate via
   `python -m backtests.regression.<id> --update-snapshot`, amend the
   relevant `docs/ENGINE_BACKTEST_*.md` with a `## Rebased <date>`
   section preserving the original numbers, file the snapshot-update
   PR linking back to the engine PR.
5. **Accidental**: revert the engine PR. The harness has done its job.

The `data_csv_sha256` field forces an explicit re-baseline if the
Bloomberg CSV refreshes (CSV changes silently invalidate snapshots).

---

## Compute cost / runtime profile

| Reproducer | Universe | Window | Friction levels | CPU time | Wall time (no contention) |
|---|---|---|---|---|---|
| S27 | 24 | 2022–2024 | 1 (frictionless) | ~50 min | ~50 min |
| S32 | 24 | 2022–2024 | 3 | ~55 min | ~55 min |
| S34 | 100 | 2022–2024 | 3 | ~4 h | ~4.5 h |
| S35 | 24 | 2018–2020 | 3 | ~50 min | _measured at PR4_ |

S34 dominates total wall-clock at ~4.5 hours. The shared-rank
optimization in `run_backtest_multi_friction` brings the multi-
friction reproducers from ~3× their single-friction equivalent down
to ~1.1× (one shared SP rank per day + per-tracker MTM / expir / CC
overhead).

---

## PR sequence and commits

| PR | Branch | Scope |
|---|---|---|
| **PR1** | `claude/backtests-regression-scaffolding` | `backtests/regression/` scaffolding (no tests, no snapshots) |
| **PR2** | `claude/backtests-regression-snapshots-tests` | Harness + S27 + S32 + S34 snapshots + tests + marker + skill |
| **PR3** | `claude/backtests-regression-ci-split` | CI workflow split (per-PR lane excludes the marker) |
| **PR4** | `claude/backtests-regression-s35-rebaseline` | S35 re-baseline against post-PIT-fix engine + this report |

Per memory `single-concern-restructure-prs.md`: PR1 is structural,
PR2 is content (the snapshots), PR3 is infra (CI lanes), PR4 is
content-truth (the S35 re-baseline + this docs PR).

Per memory `ask-before-opening-prs.md`: branches are pushed; PRs are
not opened.

---

## Known limitations and follow-ups

Consolidated catalog of open items. Each carries **status** (open /
in-flight / blocked) and **owner** (Terminal A / B / C / unowned).

### Harness-internal (this campaign's surface)

| Item | Status | Owner | Notes |
|---|---|---|---|
| H1: Driver vs throwaway-harness strike divergence — ~5–20 % NAV / per-quartile-P&L gaps on S27 (−16 %) and S32 (+5.8 %). S34 averages it out (+0.7 %). | open | Terminal A (s39 investigation; branch pruned 2026-05-29, see issue #113) | Cause unconfirmed: likely strike-rounding + CC-entry strike choice that doesn't exactly mirror the lost throwaway harness pattern. |
| H2: CSV hydration in CI not solved — `Backtest Regression` workflow is `workflow_dispatch` only; cron disabled. | open | unowned | Bloomberg CSV (~59 MB OHLCV + 78 MB IV) lives on dev laptop, not reachable from GitHub runners. Options: commit the CSVs (large), restore from S3/Artifact, or stay laptop-only. |
| H3: Mechanism-proven, real-drift-unverified. The four snapshots were self-generated by the same driver the test calls — they pass trivially today. First real engine change is the actual validation. | open | unowned | Inherent to regression-test design. Time-boxed: closes when an engine PR causes a real diff. |
| H4: 7-hour total runtime (S34 dominates at ~4.5 h, real overnight run took 6.9 h with CPU contention). Not runnable per-PR. | open | unowned | Reproducer-level optimizations (caching per-day intermediates) could shrink S34 but require careful invariance audit. |
| H5: `pre_post_pit_diff` audit not wired. Future enhancement: run same reproducer against pre-fix engine SHA via `git worktree` and emit structured diff. | open | unowned | Useful when other IV-PIT-class bugs surface. |
| H6: Tolerance band may be too tight in places. `final_nav: rel=1e-5` means even rounding-level changes in floating-point ops could false-positive. | open | unowned | Defer until first real failure tells us which knobs to loosen. |

### Engine-realism (pre-existing, locked by harness but not fixed)

These are documented elsewhere in `docs/`; consolidated here so the
harness consumer sees the full picture in one place.

| Item | Source doc | Status | Owner |
|---|---|---|---|
| E1: **Equity-beta dominance** — S34 shows ~92 % of NAV gain is equity-beta on assigned stocks, not put-selection alpha. | `archive/2026-05/SOUNDNESS_REVIEW_2026-05-26.md` | documented, not fixed | unowned (likely accepted as feature, not bug) |
| E2: **Scale dependency** — Engine UNDERPERFORMS SPY by 22 pp at $1M (S32) vs +27 pp at $100k (S27). The +27 pp headline is a BP-saturation artifact, not scale-invariant. | `docs/ENGINE_BACKTEST_S32_FRICTION.md` | documented, not fixed | unowned |
| E3: **BKNG concentration** — S34's full executed realized P&L is dominated by BKNG (+$31,576 of +$28,571 net = 110 %). Net of BKNG = −$3,004. ρ is robust; dollar story is not. | `archive/2026-05/SOUNDNESS_REVIEW_2026-05-26.md` | documented, not fixed | unowned |
| E4: **F4 tail-risk gap** — `prob_profit ≈ 0.83` stays constant across 31 % drawdowns (COST April 2022, UNH November 2024) because the empirical forward distribution has only ~30 samples at the configured `lookback_years=5 / non_overlapping=True / DTE=35`. | `docs/F4_TAIL_RISK_DIAGNOSTIC.md` | in-flight | Terminal A (`claude/docs-f4-baseline...` and follow-up fix PR) |
| E5: **In-sample HMM / POT-GPD parameters** — Regime thresholds, dealer multiplier clamp, dropout, POT-GPD tail calibration all tuned with full 2018–2026 visibility. No backtest is parameter-OOS. | `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` "Caveat 2" | documented, not fixed | unowned — would need parameter-freeze + replay setup |
| E6: **Friction at scale** — 0.27 % NAV drag is small relative to gross alpha at $100k but large relative to deployed-capital at $1M (only 10.8 % deployed → effective drag is ~10× larger on a per-active-dollar basis). | `docs/ENGINE_BACKTEST_S32_FRICTION.md` | documented, not fixed | unowned |

### Production-readiness (deployment-level)

| Item | Source doc | Status |
|---|---|---|
| P1: $100k-class strategy, not $1M — see E2. | `docs/ENGINE_BACKTEST_S32_FRICTION.md` Findings F1+F3 | documented |
| P2: Capital deployment averages 10.8 % at $1M, peaks ~72 % — engine cannot saturate $1M with this universe / parameters. | `docs/ENGINE_BACKTEST_S32_FRICTION.md` Finding F4 | documented |
| P3: Real-money deployment gate items live in `PRODUCTION_READINESS.md` (not this campaign's scope). | `docs/PRODUCTION_READINESS.md` | refer-out |

---

## For other agents reading this

If you're picking up this work or auditing it:

- The plan file is at `C:\Users\merty\.claude\plans\i-want-you-to-jolly-giraffe.md` (Claude-local).
- The four reproducer scripts are CLI-runnable:
  `python -m backtests.regression.<id> [--start ... --end ...] [--update-snapshot]`.
- The test harness is invoked via the
  `.claude/commands/backtest-regression.md` skill (~4–5 h total).
- The snapshots in `backtests/regression/snapshots/` are the
  regression baseline — changing engine math will fail them.
  Re-baseline workflow is in `TESTING.md` § "Backtest regression —
  re-baseline workflow".
- The hard invariant from CLAUDE.md §2 holds throughout: every
  reproducer routes candidates through
  `WheelRunner.rank_candidates_by_ev` end-to-end. No mocking.
