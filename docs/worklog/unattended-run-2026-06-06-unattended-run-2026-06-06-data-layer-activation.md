---
id: unattended-run-2026-06-06
title: "Unattended run 2026-06-06: data-layer activation PRs (R0a/R5/R2-R3-R6/R7)"
kind: docs
status: complete
terminal: desktop
pr:
decisions: []
date: 2026-06-06
headline: End-of-run summary of the ~9h unattended data-layer activation queue — 6 PRs opened (none merged), all CI green except the one known pre-existing f4 smoke. Deep-read ships DEFAULT-OFF; the trio is untouched beyond R0a; R1 (data merge + re-baseline + flip-on) is left for the architect-reviewed session.
surface: []
---

## Scope
Unattended queue: TASK 1 (unblock #332 + re-ship R0a), TASK 2 (R5 fingerprint),
TASK 3 (deep-read activation R2+R3+R6, default-OFF), TASK 4 (R7 hygiene + extra
coverage). All work is branch + PR, left OPEN for review. Nothing merged to main.

## PRs opened (branch · CI · does / does NOT)

| PR | Branch | What it does | CI |
|---|---|---|---|
| **#332** | `claude/deep-data-wiring-design` | **Docs-only** data-layer activation plan: verified inventory (33 refresh CSVs + 13 deep panels), R0–R7 roadmap, connector deep-read + survivorship design. R0a was split out (now #333) so this carries no trio edit. | green except pre-existing f4 |
| **#333** | `claude/fix-credit-rating-deadread` | **R0a** — `wheel_runner.py` reads `credit_rating` from the `sp_rating` key (dead-read fix) + regression test + lane claim (board #113). | green except f4 (failed test = f4 only, verified) |
| **#334** | `claude/r5-fingerprint-vol-iv-treasury` | **R5** — pin `vol_iv` + `treasury` sha in the regression snapshot fingerprint; backfilled the 4 snapshots' fingerprint (claim numbers byte-identical). | green except f4 (failed test = f4 only, verified) |
| **#335** | `claude/deep-read-activation` | **R2** connector deep-read (`_load` assembles monolith ∪ deep ∪ delisted, default-OFF) + **R3** survivorship harness (`backtests/survivorship.py`, PIT universe + delisting-aware terminal value) + **R6** Lehman-loss proof + CI-runnable synthetic assembly coverage. | non-test checks green; Test Suite expected green-except-f4 (full suite local: 2808 passed + f4) |
| **#336** | `claude/r7-deep-iv-sentinel` | **R7** — null the deep-IV `134217.7` sentinel on the assembled vol_iv read (threshold 10,000, not the roadmap's 500 — preserves real distressed IVs). **Stacked on #335** (retargeted base→main for CI). | CI triggered on retarget |

**Each PR explicitly does NOT:** merge to main; flip the deep-read default to ON;
execute R1 (the data merge) or the S27/S32/S34/S35 re-baseline; do R0b (the
sector-cap source / R9-gate change); touch the decision-layer trio beyond the R0a
one-liner.

## The one known CI failure (ignored, per the run rules)
`tests/test_f4_rv_widening.py::TestF4CasesRanker::test_calm_regime_5_ticker_smoke_preserves_main_baseline`
fails on **every** PR (and on pristine `origin/main` `efc491c`). It is a
date-sensitive no-`as_of` smoke that ages out as wall-clock (2026-06-06) advances
past the stale committed data (2026-03-20) — JPM falls outside the staleness
window, 4 rows instead of 5. **Not caused by any of these PRs** (verified: the
only failing test in #333/#334 CI is this one; #332 is docs-only). R1's fresh data
clears it. Not chased.

## Dependency stack / merge order for the architect session
1. **#332** (docs), **#333** (R0a), **#334** (R5) — independent; merge any order.
   **#334 should land before any re-baseline.**
2. **#335** (R2+R3+R6) — independent, default-OFF.
3. **#336** (R7) — **stacked on #335**; merge #335 first (then #336 collapses to
   the R7 commit).
4. **DEFERRED to the architect-reviewed session (NOT done here):** R1 (data-only
   merge of the refresh `data/` into main — take `data/` only, the branch reverts
   the engine) + the S27/S32/S34/S35 re-baseline + flipping the deep-read default
   ON (`SWE_DEEP_HISTORY` / `deep_history=True`) — these go together (the flip is a
   re-baseline event). Also deferred: **R0b** (sector-cap source = an R9-gate
   behaviour change needing a coordinated re-baseline) and the remaining R7 data
   ops (drop `sp500_short_interest.csv.xlsx`, shard `sp500_bid_ask.csv`, deprecate
   `sp500_vol_dvd.csv` — refresh-branch data files, part of R1 intake).

## Guardrails held
Nothing merged. No trio edits beyond R0a. Deep-read default-OFF on every path. No
data committed (deep-read/R3/R7 tested against a non-git scratch dir via
`SWE_DEEP_TEST_DATA`; the code PRs are code-only). Lab data branches read-only.
Primary clone's dirty tree untouched. Every task documented in `docs/worklog/`.
Per-PR detail: `fix-credit-rating-deadread`, `r5-fingerprint`,
`deep-read-activation`, `r7-deep-iv-sentinel`, `data-layer-activation` fragments.
