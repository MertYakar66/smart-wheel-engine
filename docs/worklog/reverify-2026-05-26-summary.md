---
id: reverify-2026-05-26
title: Re-verification 2026-05-26 — S1-S27 summary
kind: verification
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

## Re-verification 2026-05-26 — Summary

This section consolidates Terminal A's re-verification pass against
the current engine at `origin/main` HEAD `8a17b0b`. Each completed
S1-S27 scenario was re-run with the original setup (or the closest
faithful proxy where setup-specific harnesses were not in the
worktree). Per-scenario sub-notes live in-line under each entry
above (and under `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`
for S27). This section is the campaign-wide read.

### Scope

- **Active set (24 scenarios):** S1-S4, S7-S21, S23-S27.
- **Skipped (3):** S5 (operator-gated MCP chart), S6 (operator-gated
  + Theta lock held by another agent), S22 (archival — pre-IV-PIT-fix
  duplicate of S27).
- **Decision-layer code (`engine/ev_engine.py`,
  `engine/wheel_runner.py`, `engine/candidate_dossier.py`): NOT
  EDITED.** Re-verification is read-only against the decision layer.

### Per-scenario verdict table

| S-id | §2 holds | Verdict match | Drift > 5% | Suspect PRs | Notes |
|---|---|---|---|---|---|
| S1 | ✅ | match | no | — | Dividend bug stays fixed; `ev_raw`/`roc`/`collateral`/`hmm_regime`/`sector`/`news_n_articles` columns all surfaced |
| S2 | ✅ | match | n/a | #104 / #122 / #127 / #128 / #129 | save/load/suggest_rolls/avail_BP all shipped |
| S3 | ✅ | match | n/a | #122 / #181 | `suggest_rolls` published cols intact; `.attrs["drops"]` populated |
| S4 | ✅ | match | yes (mild) | #109 / #119 / #121 / #179 | `collateral`/`roc` columns shipped; `select_book(account_size, ..., max_weight_per_name=...)` selects 3 names at $50k |
| S5 | n/a | skipped | n/a | — | Operator-gated; no live TradingView Desktop |
| S6 | n/a | skipped | n/a | — | Operator-gated + Theta lock held by another agent |
| S7 | ✅ (by structure) | partial | n/a | — | `EngineIntegration.evaluate_trade` signature evolved; structural findings still apply at source |
| S8 | ✅ | match | n/a | #122 / #124 / #126 / #129 / **#145 (D16)** | D16 confirmed live: `EVAuthorityRefused` on neg-EV row at `issue_ev_authority_token` |
| S9 | ✅ | match | n/a | #121 | History/event/survivorship gates fail-closed; structured drops `{ticker, gate, reason}` |
| S10 | ✅ | match | n/a | #119 | News mult=1.0 silent neutral on missing store (by design) |
| S11 | ✅ | partial | yes | **#119** | `credit_multiplier` now PIT-aware (0.80 / 0.92 at 2025-04 VIX spike vs originally pinned 1.00) — PIT leak CLOSED |
| S12 | ✅ | match | n/a | — | engine_api `_enrich_alert` / ring buffer / nonce-register / HMAC all unchanged |
| S13 | ✅ | match | yes | #179 | FIX evDollars 2263.5 → 2547.97 (+12.6%); regime label `ELEVATED` + vix `28.97` bit-identical |
| S14 | ✅ | match | <5% | #118 / #220 | Layer-2 IV overlay alive (AAPL 76.97 vs orig 77.0); strangle ranker carries EventGate |
| S15 | ✅ | partial | n/a | **#163 / #165 (D17)** | 3 of 6 unwired surfaces wired; HRP + Kelly still orphan |
| S16 | ✅ | partial | yes | #179 | CAT EV +53%, NVDA magnitude -43% (both signs preserved); HMM identity holds at 4dp |
| S17 | ✅ | match | n/a | — | EV-flip + HMM-regime-flicker pattern reproduced at noise floor; zero crashes/warns |
| S18 | ✅ | **partial — WARM REGRESSION** | yes | **#215 / #220 + #208 / #210 / #222** | L2 warm 10.5s → 41.2s (+292%); HMM cache 492 → 491 (within 1) |
| S19 | ✅ — **C7b CLOSED** | partial | n/a | **#204 (R1a)** / **#215** | +inf / NaN / -inf all → `blocked / ev_non_finite`; future as_of → typed `ValueError` |
| S20 | ✅ — **G3 RE-REFUTED** | match | yes (AAPL EV) | #179 | Webhook +inf/-inf/NaN → server-computed AAPL EV/skip; ring-trim/torn-read/nonce/HMAC clean |
| S21 | ✅ | match | n/a | — | Prong A `sector_cap_breach` on CAT @ $150k; Prong B 2/9 opened at $1M, 7×`portfolio_delta_breach` |
| S22 | n/a | skipped | n/a | — | Archival pre-IV-PIT-fix |
| S23 | ✅ — **F1 + F3 CLOSED** | partial | yes | **#179 + #180** | get_recent_earnings exists; AVGO blocked at TDA; IV PIT-aware (0.4844/0.4982) |
| S24 | ✅ | match | n/a | — | take_snapshot 3-state map; `open_strangle` still absent |
| S25 | ✅ — **F3 + F4 CLOSED** | match | yes | **#179** | MU CC iv 0.6939 / 0.6515 — matches IV file exactly; sign of 25-Δ CC stays negative |
| S26 | ✅ | match | yes | #179 / #122 | MU edge $2181.54 (orig $1876, +16.3%); AAPL edge $213.59 (orig $229, -6.8%) |
| **S27** | ✅ | **partial** | yes (NAV, executed trades) | composite | ρ 0.1881 (orig 0.2183, -14%); NAV $164,876 (orig $151,444, +9%); 15 executed (orig 50, -70%). Per-year shape + quartile monotonicity preserved. Sub-note in `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`. |

### Drift report — every flagged metric

Drifts > 5% by scenario, original → new, attributed:

| Scenario | Metric | Original | New | Δ% | Suspect PR | Cause hypothesis |
|---|---|---|---|---|---|---|
| S4 | `rows[36-name watchlist]` | 30 | 27 | −10.0% | #220 + Bloomberg refresh | as_of-cutoff guard + earnings-calendar drift |
| S4 | `positive_ev rows` | 20 | 21 | +5.0% | composite (#179 + #208) | post-IV-PIT EV recalibration |
| S11 | `credit_mult[2025-04-07]` | 1.00 (pinned) | **0.80** | −20.0% (PIT-correct) | **#119** | PIT-leak fix — credit overlay now responds to as_of |
| S11 | `credit_mult[2025-04-09]` | 1.00 | **0.92** | −8.0% (PIT-correct) | **#119** | same |
| S11 | `cross-section_hmm_mean[2025-04-09 peak]` | 0.20 | 0.2923 | +46.2% | #208 + #222 | HMM disambiguation columns + seed-stable refits |
| S11 | `event_survivors[2025-04-24]` | 3 | 2 | −33.3% | #220 (low-confidence) | as_of-cutoff may drop a borderline name |
| S13 | `dashboard FIX evDollars` | 2263.5 | 2547.97 | +12.6% | **#179** | IV-PIT in `rank_candidates_by_ev` |
| S16 | `CAT ev_dollars` | 290.26 | 444.99 | +53.3% | **#179** | PIT-IV propagation; HMM mult unchanged (0.4538 / bear) |
| S16 | `NVDA ev_dollars` | −124.32 | −70.76 | −43.1% magnitude | **#179** | Same; sign preserved |
| S18 | `L2 warm wall_s` | 10.5 | 41.2 | **+292%** | **#215 + #220 + diagnostic columns (#208/#210/#222)** | Composite per-call overhead; real warm-path regression |
| S18 | `L1 cold wall_s` | 145.2 | 79.3 | −45.4% | composite (HW + cache) | Hardware/state variance + cache improvements |
| S18 | `L5b top_n=10000 survivors` | 433 | 423 | −2.3% | within noise | Bloomberg-data window drift |
| S20 | `AAPL webhook ev_dollars` | −95.47 | −14.46 | −85% magnitude | **#179** | Post-IV-PIT AAPL ev at this as_of is materially less negative; sign preserved |
| S21 | `Prong B opened/9` | 2 | 2 | 0% | — | Identical |
| S23 | `AVGO iv[2026-03-10]` | 0.4296 | 0.4844 | +12.8% | **#179** | PIT IV |
| S23 | `AVGO iv[2026-03-13]` | 0.4296 | 0.4982 | +16.0% | **#179** | PIT IV |
| S23 | `AVGO ev_dollars[2026-03-10]` | +310.30 | +390.06 | +25.7% | **#179** | Higher PIT IV → higher synthetic premium |
| S23 | `AVGO ev_dollars[2026-03-13]` | +150.46 | +222.72 | +48.0% | **#179** | same |
| S23 | `AVGO survives at 2026-03-05 (TDA)` | True (iv=0.43, ev=+268.85) | False (event_lockout) | (block) | **#180** | Symmetric event-gate back-buffer now reachable via `get_recent_earnings` |
| S25 | `MU CC iv[2026-03-17]` | 0.6485 (snapshot) | **0.6939** (PIT) | +7.0% | **#179** | Predicted post-fix; observed exactly |
| S25 | `MU CC iv[2026-03-19]` | 0.6485 | **0.6515** | +0.46% | #179 | Within rounding (snapshot vs PIT happen to coincide here) |
| S25 | `MU CC ev_dollars[2026-03-17]` (25-Δ proxy) | −1058.28 | −147.93 | −86% magnitude | #179 + strike-shift | Higher IV moved the 25-Δ strike from 541 to 557.5 |
| S26 | `MU winning edge` | +1876.05 | +2181.54 | +16.3% | **#179 + #122** | IV-PIT + buyback_total correction |
| S26 | `AAPL challenged edge` | +229.26 | +213.59 | −6.8% | within noise | Smaller grid (4 vs 16 candidates) + IV-PIT |
| S27 | `Spearman ρ` | 0.2183 | **0.1881** | −13.8% | composite (engine SHA `d26a8d6` → `8a17b0b`) | Per-year shape + quartile monotonicity preserved |
| S27 | `Final NAV` | $151,444 (+51.4%) | **$164,876 (+64.9%)** | +8.9% / +13.5 pp | mostly fewer-but-cleaner executions | Both runs $100k-class; both beat SPY +24% by 25–40 pp |
| S27 | `Executed trades` | 50 | **15** | **−70%** | composite (#215 / #220 / #227 cutoff guards + harness-shape BP gating) | Most likely contributor is harness-shape BP gating; see S27 sub-note diagnosis |
| S27 | `Mean realized (per trade)` | $63.34 | **$51.70** | −18.4% | post-#215 cutoff guards | Hit-rate rose (76.4% → 80.5%) so fewer trades pick up smaller average gains |
| S27 | `2022 mean realized` | $21.68 | **$1.72** | **−92%** | F4 tail-risk gap (open since original S27) | 2022-bear executions leaned more on losers; F4 still unresolved |

**Headline pattern.** Three PRs are responsible for ~90% of the
positive drift on the EV-magnitude axis:

1. **PR #179 (IV-PIT fix on `rank_candidates_by_ev`)** — propagated
   to `rank_covered_calls_by_ev` and `rank_strangles_by_ev` per
   PR #220's gate extension. Cited as the primary suspect on 11 of
   the flagged metrics above.
2. **PR #119 (news / credit PIT-leak fix)** — cited on the S11 credit-
   regime overlay closure.
3. **PR #180 (symmetric event-gate back-buffer via `get_recent_earnings`)**
   — closes S23 F1 (the previously-dead back-buffer is now reachable).

Plus two regressions:

4. **PR #215 + #220 (`as_of-beyond-data` refusal guards) + diagnostic
   columns from PR #208 / #210 / #222** — composite per-call overhead
   that bumps the L2 warm-rank wall-clock from ~10s to ~40s on the
   503-ticker universe. **Warm-path regression, not a §2 issue.
   Flagged. Not fixed in this PR.**

### §2 status

**GREEN.** No §2 BREACH surfaced across the 24 active scenarios.
Two §2 surface closures confirmed live:

- **S19 C7b** (dossier reviewer `+inf` bypass) — **CLOSED by PR #204**
  via R1a's `math.isfinite(ev)` check, returning the distinct
  `verdict_reason="ev_non_finite"`. Verified end-to-end with synthetic
  `ChartContext(is_ok=True, visible_price=spot)` against the EV vector
  `(+25, +inf, -inf, NaN)` — all three non-finite values return
  `blocked / ev_non_finite`. CLAUDE.md §2 R1a's text matches the
  current `engine/candidate_dossier.py:130-153` source.
- **S20 G3** (network-surface `+inf` bypass via the TV webhook) —
  **RE-REFUTED** on the v5 backfill of the original test. Webhook
  payloads carrying `ev_dollars` as `+inf` / `-inf` / `NaN` all return
  the server-computed AAPL EV (-14.46 on `2026-03-20`) with
  `verdict=skip`. The payload's `ev_dollars` is structurally never
  read on the EV path; `_enrich_alert` overrides from the ranker.

### Pre/post pytest delta

| Phase | Total | Passed | Failed | xfailed | Δ from pre |
|---|---|---|---|---|---|
| Pre-flight (baseline at `8a17b0b`) | 2394 | 2375 | 17 | 2 | — |
| Post-run (rebased onto `46ddbd4`) | **2417** | **2412** | **3** | **2** | **+14 passed, −14 failed** |

**The 14-test improvement is not from this re-verification work.**
It landed via **PR #237** (`fix(tests): extend synthetic OHLCV to
cover as_of=2026-03-15`), which merged into `main` during my work
and was pulled in via `git rebase origin/main` before the final
pytest run. PR #237 closes the `test_ranker_iv_pit.py` (9 fails) and
`test_event_gate_back_buffer.py` (5 fails) clusters by extending the
mock connectors' synthetic OHLCV ranges so the new
`as_of-beyond-data` cutoff guards (PR #215 / #220) don't refuse the
test fixtures.

The **3 remaining post-run failures** are all in
`tests/test_theta_connector.py` — Windows-local per the
[[windows-local-vs-ubuntu-ci]] memory; not extrapolable to CI;
present on `main` before this work.

**Conclusion: no regression introduced by this re-verification.**
Pre-existing failures: 17 → 3 (improvement attributable to PR #237).
My work contributes 0 new failures.

The total test count went 2394 → 2417 (+23) because PRs that
merged into main during my work added regression tests
(e.g. PR #233's 13 D17-wire regression tests).

### 5-ticker EV smoke drift

| Phase | sha256 | rows | cols | Top by EV |
|---|---|---|---|---|
| Pre-flight (at `8a17b0b`) | `4fc14bf0e6985ac42fe9f9f04352df8884e2c0e51bdcf52bc08626e7905c5317` | 5 | 51 | XOM ($137.57), JPM (124.90), MSFT (90.97), UNH (62.62), AAPL (20.45) |
| Post-run (at `46ddbd4`) | `4fc14bf0e6985ac42fe9f9f04352df8884e2c0e51bdcf52bc08626e7905c5317` | 5 | 51 | (identical) |

**Result: IDENTICAL.** Byte-for-byte parquet match. The 5-ticker
EV smoke is unchanged across the rebase + re-verification work,
confirming no in-process state mutation leaked from re-verification
into the live data path. Saved at `C:\tmp\preflight_5tk.parquet`
and `C:\tmp\postrun_rebased_5tk.parquet`.

Pre-flight saved to `C:\tmp\preflight_5tk.parquet` (not in repo).
Post-run will be diffed byte-for-byte against this; any deviation
indicates an unintended state mutation during re-verification.

### Methodology / setup substitutions

- **S17 condensed**: ran a 5-day rolling rank sweep on the documented
  25-ticker universe instead of the full 10-trading-day operational
  sim (with daily save/load round-trips). The condensed run exercises
  the same per-day EV-flip and HMM-flicker mechanism. The "operational
  verdict: YES with workarounds" outcome from the original entry is
  preserved by reference, not re-derived end-to-end.
- **S20 ports**: ran the engine_api subprocess on Terminal A's
  allocated port `:8787` instead of S20's documented `:18787` (which
  is outside Terminal A's `.claude/settings.local.json` env block).
  No behavioral difference — same code path, same race vectors.
- **S22**: skipped per task spec (archival).
- **S27**: run via a one-off harness modeled on Terminal C's unmerged
  `backtests/regression/_common.py` (read-only reference; not imported
  or committed into this re-verification PR). Setup matches the doc
  exactly: 2022-01-03 → 2024-12-31, 24 tickers, $100k, 35-DTE / 25-Δ,
  frictionless, `require_ev_authority=False`.

### What was NOT re-verified

- **Theta-provider-specific paths** (S6 and the chain-quality gate at
  `wheel_runner.py:843`). Theta Terminal access is held by another
  agent for the duration of this work — no contention.
- **MCP live chart loop** (S5). Requires TradingView Desktop + CDP
  on `:9222` + tradingview-mcp CLI.
- **Performance regression follow-up on S18 L2 warm path.** Captured
  the drift; identification of the root-cause PR is composite and
  needs profiler-level work to attribute precisely.
- **Handle-leak follow-up from S18.** Would require a 100+-call
  sweep; the warm-path regression is more pressing.

### Recommended follow-ups (not in scope for this PR)

1. **S18 warm-path regression.** Profile a single warm
   `rank_candidates_by_ev(503-ticker)` call to identify which gate-
   check or diagnostic-column emit is responsible for the 4× warm
   latency. Plausible suspects (in priority order): PR #215's
   `_check_as_of_cutoff` running per-ticker, PR #220's CC/strangle-
   ranker variants of the same guard, PR #210's `sector` column
   needing a per-ticker `get_fundamentals` lookup, PR #222's HMM-
   disambiguation columns adding per-call regime fetches.
2. **`tests/test_ranker_iv_pit.py` + `tests/test_event_gate_back_buffer.py`
   mock fixtures.** 14 of the 17 pytest failures stem from mock
   connectors that don't expose the data-cutoff or recent-earnings
   metadata the new guards require. Update the mocks to match the
   real connector's API surface.
3. **S7 advisor committee re-test under the new `evaluate_trade(ev_row,
   portfolio_state, market_state)` signature.** The "committee
   structurally pinned at neutral" finding wasn't re-derived; a small
   Sn that exercises the new shape on the 10 ROC-ranked names would
   close the loop.

### Engine state — overall posture

The engine is **mechanically sound on the §2 invariant** at
`origin/main` HEAD `8a17b0b`. The campaign-headline closures from
the original entries (S19 C7b inf-bypass, S23 F1 back-buffer dead
code, S23 F3 / S25 F3 IV-snapshot bug across all three rankers,
S11 credit-PIT leak) have all shipped. Drift > 5% is concentrated
on EV-magnitude axes where PR #179's IV-PIT propagation legitimately
shifts the engine's view of the world; signs and orderings are
preserved.

The most material unresolved item this re-verification surfaces is
the **S18 warm-rank latency regression** — not a §2 issue, but a
real operator-facing throughput change worth a dedicated follow-up.
