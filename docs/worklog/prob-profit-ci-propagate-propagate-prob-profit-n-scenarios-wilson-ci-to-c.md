---
id: prob-profit-ci-propagate
title: Propagate prob_profit n_scenarios + Wilson CI to CC/strangle rankers, engine_api, and the Ollama trade memo
kind: feature
status: in-flight
terminal: UltraCode
pr:
decisions: []
date: 2026-06-01
headline: Follow-up to the prob_profit Wilson-CI honesty fix — extends n_scenarios + the Wilson 95% CI from the put ranker to ALL consumers a trader actually reads: the covered-call + strangle rankers, the engine_api HTTP surface (/api/candidates camelCase + /api/tv/dossier ev_row pass-through), and the Ollama trade memo. Strictly additive; prob_profit/EV/verdict unchanged. Built by 3 file-disjoint agents in one shared worktree; verified by 207 passing tests on the integrated diff.
surface:
  - engine/wheel_runner.py
  - engine_api.py
  - engine/trade_memo.py
  - tests/test_covered_call_ranker.py
  - tests/test_strangle_ev_ranker.py
  - tests/test_tv_api.py
  - tests/test_trade_memo_ci.py
---

## Goal
<!-- What we set out to do, and why. -->

The base fix (`claude/prob-profit-ci-honesty`) surfaced `prob_profit`'s
small-sample Wilson 95% CI + N on `EVResult` and the PUT ranker. But the
audit found the over-confident bare number still reached the trader
unqualified on every OTHER surface: the covered-call/strangle rankers, the
`engine_api` HTTP responses (`/api/candidates` ships `probProfit` raw,
`probability=prob_profit*100`), and the Ollama trade memo (no uncertainty
language at all). This card extends the honesty to all of them so the
caveat travels wherever `prob_profit` does.

## What we tried
<!-- Approaches, in the order we tried them. -->

Fanned out 3 file-disjoint agents in one shared worktree
(`claude/prob-profit-ci-propagate`, stacked on the base CI branch), one per
surface: (1) `wheel_runner` CC + strangle rankers, (2) `engine_api`, (3)
`trade_memo`. Each grounded its surface, implemented additively, added a
test, and ran it. I then verified the *integrated* diff against source and
re-ran the affected suite myself (subagents drift).

## What worked

All three surfaces, additive, verified. The agents respected file ownership
(no cross-file edits) and the EVResult field names.

## What didn't
<!-- The dead ends + WHY. This is the part that saves the next agent. -->

Nothing dead-ended. One sharp edge the CC/strangle agent had to handle (and
I pre-flagged): UNLIKE the put ranker (`pd.DataFrame(rows)`), the CC and
strangle rankers reindex to PINNED column lists
(`_CC_RANK_CORE_COLUMNS` / `_STRANGLE_RANK_DIAGNOSTIC_COLUMNS`), so the new
keys had to be added to BOTH the column constant AND the row dict or pandas
silently drops them (and the `list(df.columns) == _..._COLUMNS` tests would
catch it). Done correctly.

## How we fixed it
<!-- The approach that shipped. -->

- **`engine/wheel_runner.py`** — CC ranker (`rank_covered_calls_by_ev`):
  `n_scenarios` + `prob_profit_ci_low/high` added to `_CC_RANK_CORE_COLUMNS`
  (right after `prob_profit`) AND the row dict. Strangle ranker
  (`rank_strangles_by_ev`, prob_profit is per-leg): a shared `n_scenarios`
  (both legs walk the same forward path) + per-leg
  `put_/call_prob_profit_ci_low/high` in `_STRANGLE_RANK_DIAGNOSTIC_COLUMNS`
  + the diagnostic row. Same NaN→None idiom as the put ranker.
- **`engine_api.py`** — `/api/candidates`: `nScenarios` / `probProfitCiLow`
  / `probProfitCiHigh` (camelCase) added after `probProfit`, pass-throughs
  of the ranker columns; a new `_nan_to_none` helper normalises the
  pandas-float64-coercion-of-None edge case to JSON null. `/api/tv/dossier`:
  no code change — the CI rides through `ev_row` (snake_case) automatically.
- **`engine/trade_memo.py`** — `_format_prob_profit_line(row)` renders
  `PROB PROFIT: 0.86 [0.71, 0.94] (N=35)` + a one-line small-sample caveat;
  `_get_ev_row` fetches the row via `rank_candidates_by_ev` (routes through
  `EVEngine.evaluate`, display-only); appended as a `--- PROBABILITY OF
  PROFIT ---` block, omitted when not evaluated.

§2: strictly additive across all surfaces — `prob_profit`, `probability`,
`ev_dollars`, `ev_raw`, the verdict, the dealer-multiplier clamp, and the
`/api/candidates` recommendation logic (`ev_dollars` + `prob_profit>=0.65`)
are all byte-for-byte unchanged. The CI is emitted/displayed, consumed by no
decision. `wheel_runner.py` (CC/strangle) is trio → lane-claim + §2 read.

## Evidence
<!-- Exact commands run, numbers, links to raw artifacts. -->

- Integrated-diff verification (mine, not the agents'):
  `pytest test_covered_call_ranker + test_strangle_ev_ranker + test_tv_api +
  test_tv_dossier + test_trade_memo_ci + test_prob_profit_ci +
  test_audit_invariants + test_dossier_invariant` → **207 passed**.
- Per-surface (agents): CC/strangle 82 passed (real AAPL survivor tests
  passed, not skipped); tv_api 13 + 160 regression; trade_memo 8.
- `ruff check` + `ruff format --check` clean on all changed files.

## Unresolved / handoff
<!-- What's still open; what the next agent should look at next. -->

- **Stacked on the unmerged base** `claude/prob-profit-ci-honesty` (the
  EVResult fields live there). Merge the base first, then this rebases clean.
- **Dashboard render** — replace the cockpit's hardcoded `TOP_BIN_REALIZED`
  TS constant with the engine-emitted CI (`probProfitCiLow/High` now on
  `/api/candidates`). Off-trio TS follow-up.
- `_get_ev_row` adds one `rank_candidates_by_ev` call (~2s) per
  `generate_memo` — acceptable for an on-demand memo; noted.
- Calibration band (the bias correction) remains the operator's gated §2 call.
