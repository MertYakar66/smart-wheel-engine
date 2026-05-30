---
id: MP-C
title: Rebase §2 pair PR #248 + #249 + CC schema fix
kind: fix
status: in-flight
terminal: c
pr: 248, 249
decisions: [D18]
date: 2026-05-30
headline: Rebased the §2 stacked pair onto post-#285 main; closed the CC schema drop that silently lost pnl_p25/50/75 on covered-call rows.
surface:
  - engine/ev_engine.py
  - engine/wheel_runner.py
  - engine/news_sentiment.py
  - engine_api.py
  - DECISIONS.md
  - tests/test_covered_call_ranker.py
  - tests/test_ev_engine_percentiles.py
  - tests/test_news_severance.py
---

## Goal

Card MP-C from the Major Session's 2026-05-29 GO comment on #113. Two §2-surface PRs from D's pre-#285 news-architecture campaign were stale against the new base (`main@482bc79` after #285 and #286 landed) and needed:

1. **#248** (`pnl_p25/50/75` distribution-spread fields on `EVResult`) — rebase + one schema fix.
2. **#249** (sever verbal news from the EV path, ships D18) — rebase onto rebased #248.

Both PRs touch the decision-layer trio, so each PR body needed a `lane-claim` block to satisfy `scripts/check_lane_claim.py` (the post-#285 CI gate). The shared `wheel_runner.py` surface is why both ended up on one terminal.

## What we tried

1. **Re-baselined to `origin/main`** in this worktree. Confirmed it's at `482bc79` exactly — the rebase target the GO comment named.
2. **Rebased `claude/lucid-davinci-pm15H-pct-spread` onto `origin/main`.** Two commits replayed without conflict. The merge-base was `70fdb78` — 12 PRs of drift, none touching `EVResult` / `_CC_RANK_CORE_COLUMNS`, which is why the replay was clean.
3. **Reproduced the #248 P2 bug by reading.** `rank_covered_calls_by_ev` builds its survivor frame with `pd.DataFrame(rows, columns=cols)` where `cols = list(_CC_RANK_CORE_COLUMNS)` (+ diagnostics). The row dict at L2538–2554 writes `pnl_p25/50/75`, but those keys aren't in `_CC_RANK_CORE_COLUMNS` so pandas drops them silently on CC rows. Put-side rows survive only because `rank_candidates_by_ev` builds its frame without `columns=`.
4. **Two fix shapes considered:** (a) add the three keys to `_CC_RANK_CORE_COLUMNS`; (b) drop `columns=cols` and let pandas infer. Chose (a) — the pinned schema is what makes `_empty()` return a same-shaped zero-row frame and what the `TestSchema` tests assert against. Dropping `columns=` would silently un-pin the shape contract.
5. **Rebased `claude/lucid-davinci-pm15H-sever-news` onto rebased #248** with `git rebase --onto`. One conflict: a meaningless three-way text overlap inside `engine/wheel_runner.py` between #248's new percentile read and a docstring expansion that landed in main; resolved by keeping the percentile read intact. D18 in `DECISIONS.md` carries through unchanged — main still tops at D17 so the slot is free.

## What worked

- **Rebase is clean per branch.** No semantic conflicts in either replay; the post-#285 main shifted nothing that #248 / #249 depended on.
- **CC schema fix is three lines.** Insert `pnl_p25 / pnl_p50 / pnl_p75` between `ev_per_day` and `prob_profit` so the schema order matches the row-dict write order at L2549–2554.
- **The existing schema-equality test at `tests/test_covered_call_ranker.py::TestSchema::test_columns_with_diagnostics` automatically validates the additions.** `list(df.columns) == _CC_RANK_CORE_COLUMNS + _CC_RANK_DIAGNOSTIC_COLUMNS` updates with the schema variable; nothing more needed for column presence.
- **Added one targeted content regression test** (`test_pnl_percentiles_reach_survivor_rows`) that asserts the three columns are non-null on survivor rows AND that `p25 ≤ p50 ≤ p75` holds. Pins the row-dict → DataFrame link so a future regression that strips one of the writes (or one of the schema entries) trips here, even if the columns themselves survive.
- **D18 stays intact.** D8 / D17 are the prior occupied slots; D18 is the next free, exactly as #249's commit message claimed.

## What didn't

- **Did not edit `USAGE_TEST_LEDGER.md`.** The post-#285 worklog redesign froze it. The campaign log lives in this fragment instead.
- **Did not retarget #249's PR base to point at #248's branch.** GitHub's stacked-PR detection auto-collapses #249's diff against `main` to just the #249-specific commits once #248 lands. Retargeting would create a temporary "merge #249 → #248" link that has to be flipped back at merge time and is more confusing than it's worth.
- **Did not attempt the "drop `columns=cols`" alternative.** Even though it'd close the immediate bug, it weakens the schema contract that `_empty()` and the three `TestSchema` tests rely on. The minimal change is the right shape here.

## How we fixed it

### #248 — pct-spread

1. `git checkout claude/lucid-davinci-pm15H-pct-spread`
2. `git rebase origin/main` (two commits replayed clean onto `482bc79`).
3. Added `pnl_p25`, `pnl_p50`, `pnl_p75` to `_CC_RANK_CORE_COLUMNS` (between `ev_per_day` and `prob_profit`, matching row-dict ordering).
4. Added regression test `TestSchema::test_pnl_percentiles_reach_survivor_rows`.
5. New commit `fix(wheel_runner): add pnl_p25/50/75 to CC schema (#248 P2)`.
6. Added a `lane-claim` block in the PR body naming `engine/ev_engine.py` + `engine/wheel_runner.py`.
7. `git push --force-with-lease`.

### #249 — sever-news

1. `git checkout claude/lucid-davinci-pm15H-sever-news`
2. `git rebase --onto claude/lucid-davinci-pm15H-pct-spread <old-base>` so the news-severance commit lands on top of the rebased #248 head.
3. Verified D18 is still in `DECISIONS.md` after the replay (no edits needed — the commit carries the full D18 block).
4. Added a `lane-claim` block in the PR body naming `engine/wheel_runner.py` + `engine/news_sentiment.py`.
5. `git push --force-with-lease`.

### Shared cleanup

- `python scripts/sync_manifest.py --fix` to bring `FILE_MANIFEST.md` up to date for this fragment + any other tracked-but-uncatalogued files (post-#279, the `--fix` mode is idempotent and only ever appends).
- `python scripts/gen_worklog_index.py` to refresh `docs/worklog/INDEX.md` (CI fails if stale per #285).

## Evidence

| Check | Command | Result |
|---|---|---|
| Targeted CC + percentile tests | `py -3.12 -m pytest tests/test_covered_call_ranker.py tests/test_ev_engine_percentiles.py -v` | **43 passed** in 1.91s |
| Decision-layer invariant set | `py -3.12 -m pytest tests/test_launch_blockers.py tests/test_audit_invariants.py tests/test_authority_hardening.py tests/test_dossier_invariant.py tests/test_ev_engine_upgrades.py tests/test_ranker_transparency.py tests/test_audit_viii_unit_invariants.py tests/test_audit_viii_e2e.py tests/test_audit_viii_real_data_smoke.py tests/test_ev_engine_percentiles.py tests/test_event_gate.py tests/test_tv_dossier.py tests/test_tv_api.py tests/test_covered_call_ranker.py tests/test_diagnostic_column_honesty.py tests/test_ranker_iv_pit.py tests/test_suggest_rolls_drops.py tests/test_pit_leaks.py` | **342 passed**, 2 warnings in 83.77s |
| Ruff format check | `py -3.12 -m ruff format --check engine/wheel_runner.py tests/test_covered_call_ranker.py` | 2 files already formatted |
| Ruff lint | `py -3.12 -m ruff check engine/wheel_runner.py tests/test_covered_call_ranker.py` | All checks passed |

## Unresolved / handoff

- **Strangle ranker (`rank_strangles_by_ev`) does NOT carry `pnl_p25/50/75`** — deliberately so, per #248's commit message: a strangle aggregates two `EVResult`s and the combined-leg percentile isn't a sum (`E[p25] ≠ p25 of E`). If a future change wants to surface a joint distribution spread for strangles, that's its own PR; `_STRANGLE_RANK_CORE_COLUMNS` is intentionally left alone here.
- **The `news_multiplier = 1.0` column on candidate rows after #249 lands** is a uniform 1.0. #249's body notes this as cleanup for the dashboard-pane PR (PR8 of the original campaign). Not in MP-C scope.
- **Backtest re-baseline after #249.** Every prior backtest with non-1.0 news multipliers becomes obsolete. Lives in PR7/9 of the original campaign (Terminal-D backlog), not MP-C.
- **D3 SUPERSEDED marker.** #249's commit explicitly flags that auto-mode classifier blocks editing D3 to add `**SUPERSEDED by D18 (verbal-news clause only)**`. D18's body carries the back-reference inline instead. If the operator wants the D3-side marker, it's a manual edit (carved out from the classifier rule); MP-C does not work around it.
