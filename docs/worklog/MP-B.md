---
id: MP-B
title: MP-B — rebase PR #251 + resolve 3 codex review threads
kind: fix
status: ready-for-merge
terminal: B
pr: 251
decisions: []
date: 2026-05-30
headline: EDGAR earnings PR rebased onto main@482bc79; codex P1 (manifest) + 2× P2 (project loop, refresh overwrite) closed.
surface: [engine/external_data/edgar_adapter.py, scripts/pull_edgar_earnings.py, tests/test_external_data_edgar.py, FILE_MANIFEST.md, docs/worklog/MP-B.md]
---

## Goal

Make PR #251 (EDGAR earnings data layer, campaign PR3/9) ready for merge under
the post-#285 conventions: rebase onto current `main`, resolve all three open
codex review threads, drop a worklog fragment in place of the deprecated
USAGE_TEST_LEDGER entry.

## What we tried

1. `git checkout claude/lucid-davinci-pm15H-edgar && git rebase origin/main`
   in the per-terminal worktree. Branch was 1 commit ahead of `b2cce25`;
   `main` was at `482bc79`. Clean rebase — no conflicts.
2. Inspected the three codex threads on PR #251 directly via the GitHub API
   (the inline review surface is the source of truth for what blocks merge).
3. For each thread, traced the cited line to the live code and confirmed the
   failure mode is reproducible from the report alone — no speculation
   needed to design the fix.

## What worked

Three independently-scoped fixes; one regression test each. Pulling the
merge logic out of `main()` into a pure helper kept the test footprint
small (no subprocess / tempdir / parquet IO in the unit test) without
expanding the script's surface.

## What didn't

First pass on the `--refresh` regression test asserted the wrong semantic
— that successfully-refreshed MSFT should KEEP its old accession alongside
the new one. That's the per-accession-merge reading of the codex hint, but
the production reality is that the SEC `submissions/<cik>.json` `recent`
block IS the authoritative state for any ticker we successfully re-pulled
(≈1000 filings, >10y of 8-Ks). Per-accession merge would preserve stale
rows the SEC has actually retracted. Corrected: "refresh = the new pull
fully replaces the prior rows for any ticker present in `new_df`; rows
for absent tickers (failed / empty refresh) are preserved." Tests now
pass that semantic explicitly.

## How we fixed it

**P1 — FILE_MANIFEST.md missing entries (red CI).** `python scripts/sync_manifest.py
--fix` added the three new tracked paths (`docs/EDGAR_EARNINGS.md`,
`scripts/pull_edgar_earnings.py`, `docs/worklog/MP-B.md`).

**P2 — `project_next_earnings` zero-delta loop.** In
`engine/external_data/edgar_adapter.py`, dedupe `hist` on `filing_date`
(`keep="last"`) BEFORE the inter-filing-delta median, and re-apply the
`min_history` floor to the deduped count. Belt-and-suspenders: after
computing `median_days`, return `None` if it's `<= 0` (shouldn't happen
post-dedupe, but a malformed history pair could in principle still
produce a non-positive median, and a defensive guard is cheap insurance
against an infinite roll-forward). `n_history` now reflects unique
filing-date count, which is the metric that actually fed the projection.

**P2 — `--refresh` overwrite.** In `scripts/pull_edgar_earnings.py`,
extracted the merge into a new `merge_with_existing(old_df, new_df, *,
refresh)` helper, then wired `main()` to call it on the
`out_path.exists()` branch regardless of `--refresh`. Refresh-mode logic:
build the set of successfully-refreshed tickers from `new_df`, drop only
those tickers' rows from `old_df`, then concat + dedupe on `(ticker,
accession)`. Tickers that were on the refresh list but failed or returned
empty are absent from `new_df`, so their prior rows survive — a partial
refresh can never silently destroy data the operator hasn't replaced.

## Evidence

```bash
# Rebase
$ git rebase origin/main
Successfully rebased and updated refs/heads/claude/lucid-davinci-pm15H-edgar.

# Targeted test pass — full edgar test file, includes the 3 new regressions
$ python -m pytest tests/test_external_data_edgar.py -v --no-cov
...
TestProjectNextEarnings::test_same_date_filings_do_not_stall_projection PASSED
TestMergeWithExisting::test_refresh_preserves_other_tickers PASSED
TestMergeWithExisting::test_refresh_failed_ticker_keeps_prior_rows PASSED
TestMergeWithExisting::test_append_mode_dedupes_on_ticker_accession PASSED
============================= 41 passed in 0.72s ==============================
```

Codex thread mapping (PR #251 inline review by `chatgpt-codex-connector`,
2026-05-27T14:04Z, reviewing commit `06856a2a`):

| Thread | File:line | Fix |
|---|---|---|
| P1 — manifest missing | `docs/EDGAR_EARNINGS.md:1` | `sync_manifest.py --fix` |
| P2 — zero-day loop | `engine/external_data/edgar_adapter.py:304` | dedupe + median guard |
| P2 — refresh overwrite | `scripts/pull_edgar_earnings.py:200` | `merge_with_existing` helper |

## Unresolved / handoff

- Puller live-verification against SEC still pending (already noted in
  `docs/EDGAR_EARNINGS.md` §"first-pull" + the PR body's "Unresolved" —
  unchanged by this rebase).
- Follow-up PR3.5 (wire `project_next_earnings` into
  `MarketDataConnector.get_next_earnings`) is §2-surface and stays on the
  campaign tracker; not in scope here.
