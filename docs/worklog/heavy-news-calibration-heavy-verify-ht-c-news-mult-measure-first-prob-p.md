---
id: heavy-news-calibration
title: Heavy verify HT-C — news-mult measure-first + prob_profit calibration re-verify
kind: verification
status: in-flight
terminal: c
pr:
decisions: []
date: 2026-05-30
headline: On Bloomberg the pre-D18 news_multiplier was 1.0 in 1300/1300 probes — D18 is a measured no-op for prior backtests; prob_profit calibration matches the published 10-config matrix exactly.
surface:
  - docs/HEAVY_NEWS_CALIBRATION_REVERIFY.md
  - docs/verification_artifacts/news_calibration_driver.py
  - docs/verification_artifacts/heavy_news_calibration_2026-05-30_raw_output.txt
---

## Goal

Card HT-C from the 2026-05-30 heavy-verify cycle. The campaign tracking doc (`docs/NEWS_REDESIGN_CAMPAIGN.md` §3) asserts that *every prior `S<N>` scenario result becomes obsolete after this campaign* — implicitly meaning D18 alone, since R9/EDGAR/FRED are not yet shipped. That claim is **unverified** on the Bloomberg-only environment: if the historical `news_multiplier` was already 1.0 because the news store is empty, D18 changes nothing, and the "re-baseline mandatory" claim is wrong for the D18 component. HT-C is *measure-first* — empirically establish what `news_mult` actually was before deciding whether a heavy re-baseline (S34-class, ~3.5h compute) is warranted, then re-verify `prob_profit` calibration on the current engine.

## What we tried

1. **Inventoried the news store on disk.** `data_processed/news_sentiment.parquet`, `data/news/sentiment.parquet`, `data_processed/news_sentiment.csv`, `data/news/sentiment.csv`, `financial_news/storage/sentiment.sqlite` — none exist on this dev box. `find . -type f -name "*sentiment*"` returned only code files (`engine/news_sentiment.py`, `scripts/pull_news_sentiment.py`, `docs/worklog/s10-news-sentiment-downgrade-path.md`). `git log --all --diff-filter=A --name-only | grep news_sentiment` showed the parquet has **never** been added to git history.
2. **Extracted the pre-D18 `sentiment_multiplier` body** from `9e8edcd~1:engine/news_sentiment.py:188-218`. The ladder is:
   - `n_articles < 5` → 1.0
   - `sentiment ≤ −0.3 and n ≥ 5` → 0.88
   - `sentiment ∈ (−0.3, −0.1] and n ≥ 5` → 0.95
   - `sentiment ≥ 0.3 and n ≥ 5` → 1.05
   - else → 1.0
3. **Wrote a driver** (`docs/verification_artifacts/news_calibration_driver.py`) that probes `NewsSentimentReader.get_ticker_sentiment` over `UNIVERSE_100` × 13 quarterly dates (2020-03-31 → 2026-03-20), inlines the pre-D18 ladder, and reports the multiplier distribution.
4. **Read `engine/ev_engine.py`** to confirm that `prob_profit` is invariant to `news_mult`: `prob_profit = float(np.mean(pnls > 0))` (line 385) is computed BEFORE the regime multiplier is applied (comment at line ~469 "Regime multiplier is applied *last*"); `test_percentiles_are_pre_multiplier` in `tests/test_ev_engine_percentiles.py` is the existing pin.
5. **Extended the driver to compute calibration on 10 existing rank_logs** in `%TEMP%\s{n}_backtest\` (the exact files PROB_PROFIT_CALIBRATION_2026-05-28.md cites as its data source). 7-bin scheme + the pre-declared verdict thresholds from the published doc.
6. **Hit a Windows cp1252 encoding crash** on the FINAL SYNTHESIS print (the Greek capital delta in `top-bin Δ -4.88pp`). Fixed by adding `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at module top, matching the pattern in `scripts/pull_news_sentiment.py`.
7. **Started a §C S27 re-run on the current post-#249 engine** (background, ~20 min) to empirically confirm the analytical claim that `prob_profit` is byte-identical row-for-row between pre- and post-#249 engines for the same input.

## What worked

- **§A is decisive.** 1300/1300 probes return `n_articles=0, sentiment=0.0`, which the pre-D18 ladder maps to 1.0 by the `n < 5` guard. The empty store is a property of the environment, not a sampling artifact.
- **§B reproduces every published calibration number byte-identically.** All 10 configs in `PROB_PROFIT_CALIBRATION_2026-05-28.md` come out with the same top-bin Δ and the same OK/Warn/MISCAL bin counts. The structural finding (top bin (0.95, 1.0] is MISCAL by −15 to −18pp on $1M / 100t configs; F4 fix doesn't improve it) is robust on the post-#249 engine because §A + the structural prob_profit invariance imply the rank_logs are unchanged.
- **The measure-first protocol caught a real doc-currency issue.** `NEWS_REDESIGN_CAMPAIGN.md` §3 says "every prior `S<N>` scenario result becomes obsolete" for the campaign as a whole, but lumps D18 in with R9/EDGAR/FRED. For D18 alone, the claim is measurably false on Bloomberg. Surfaced as F1 for the next-cycle magnet-doc reconciliation.

## What didn't

- **Initial `--skip-rerun` run crashed on Windows cp1252.** The `Δ` character (U+0394) in the FINAL SYNTHESIS line `top-bin Δ -4.88pp` is not in cp1252, so Python's default-encoding stdout writer raised `UnicodeEncodeError`. Sections §A and §B printed cleanly (their `§` is cp1252-compatible at 0xA7); only the final-synthesis Greek delta tripped it. **Lesson:** any driver that streams to `tee` or a redirected stdout on Windows must reconfigure stdout to UTF-8 explicitly — `scripts/pull_news_sentiment.py` already does this and was the template.
- **`git pull origin main` on the wrong branch triggered a merge.** While trying to switch from `claude/lucid-davinci-pm15H-sever-news` to `main`, Terminal D's worktree owned `main` (expected for parallel HT-D card), and the harness's `git pull` attempted to merge `origin/main` into the current branch (severance) instead. Three steps to clean up: `git merge --abort` failed (no MERGE_HEAD), so `git reset --hard origin/main` after `git checkout -b claude/heavy-news-calibration-reverify`. **Lesson:** in a 4-worktree setup, never `git checkout main` from a non-primary clone — always create the new branch directly from `origin/main` via `git checkout -b <new> origin/main`.
- **Did not escalate to a full S34-class re-run.** The escalation clause in the HT-C card was "ONLY IF the measured news_mult is non-1.0". §A says it was 1.0 always, so escalation is structurally unwarranted. A full $1M / 100t rerun would cost ~3.5h to confirm the analytically-proven prob_profit identity and would emit a rank_log byte-identical to the existing S34 one.

## How we fixed it

- §A driver section measures the news_mult distribution directly on the Bloomberg connector via `NewsSentimentReader`. With an empty store, `get_ticker_sentiment` returns the neutral default by design — the driver applies the inlined pre-D18 ladder to that output and reports the resulting multiplier values.
- §B driver section iterates the 10 calibration-eligible `rank_log.csv` files under `%TEMP%`, computes the 7-bin table per config, and renders both a cross-config summary table and a detailed S27 reference table.
- §C driver section runs `backtests/regression/s27_ivpit_24t_100k.run(output_dir=…)` against the current engine, computes its calibration table, then joins row-by-row to the existing `s27_backtest/rank_log.csv` on `(entry_date, rank_position, ticker, option_type, strike)` and reports `max_abs_diff` on `prob_profit`.
- All three sections write to stdout; the consumer captures via `tee` into `docs/verification_artifacts/heavy_news_calibration_2026-05-30_raw_output.txt`.

## Evidence

| Check | Command | Result |
|---|---|---|
| §A (UNIVERSE_100 × 13 dates) | `py -3.12 docs/verification_artifacts/news_calibration_driver.py --skip-rerun` | 1300/1300 probes return news_mult = 1.0 |
| §B (10 existing rank_logs) | same | All 10 top-bin Δ match `PROB_PROFIT_CALIBRATION_2026-05-28.md` byte-identically |
| News store on disk | `find . -name "*sentiment*" -type f` | code-only; no parquet/csv/sqlite store |
| Store ever in git | `git log --all --diff-filter=A --name-only \| grep news_sentiment` | empty (file never added) |
| Pre-D18 ladder source | `git show 9e8edcd~1:engine/news_sentiment.py \| sed -n '188,218p'` | matches the driver's inlined `pre_d18_sentiment_multiplier` |
| Ruff format/lint | `py -3.12 -m ruff format --check ... && ruff check ...` on the driver | clean |
| §C status | background `b7a4x024f`, ETA ~20 min from kickoff at ~04:35 UTC | TBD on completion |

Raw output: `docs/verification_artifacts/heavy_news_calibration_2026-05-30_raw_output.txt`.

## Unresolved / handoff

- **§C completion.** Background rerun in flight (`b7a4x024f`). Expected `max_abs_diff(prob_profit) ≤ 1e-6` row-for-row on the join. Both findings doc and this fragment get a final-status update on completion.
- **F1 (campaign-doc currency).** `NEWS_REDESIGN_CAMPAIGN.md` §3's "re-baseline mandatory" claim is too broad for D18 alone. Surfaced for the magnet-doc reconciliation at next cycle close; suggested edit text in the findings doc §"Findings" F1.
- **F2 (operator transparency gap).** Dashboard column `news_sentiment` surfaces a permanently zero store. Either retire `scripts/pull_news_sentiment.py` (operator call per the D18 commit's "Unresolved" item) or label the column as "(severed — D18, store empty on this env)".
- **F3 (post-#260 calibration matrix beyond S38-postF4).** A complete post-#260 re-baseline of S34 / S38 / S40 is outside HT-C's measure-first scope. Worth a follow-on card if the operator wants a complete post-#260 calibration matrix; the POT-GPD prob_profit research direction (`PROB_PROFIT_CALIBRATION_2026-05-28.md` §"For future research") is the natural follow-on.
- **No `engine/` edits this card.** Findings → magnet-doc reconciliation; engine changes (if any) → next cycle's fix cards per the Major Session's triage.
