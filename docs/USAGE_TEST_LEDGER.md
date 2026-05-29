# Usage Test Ledger — FROZEN (migrated to docs/worklog/)

> **This file is frozen.** Its `S1`–`S46` usage-test scenarios were split,
> verbatim, into per-task fragments under `docs/worklog/` on 2026-05-29 (the
> documentation redesign — `DECISIONS.md` D14 extension). **Do not add or edit
> entries here.** Create a fragment with `python scripts/new_worklog.py` and let
> `scripts/gen_worklog_index.py` index it. The live index is
> [`docs/worklog/INDEX.md`](worklog/INDEX.md).
>
> **Why:** this file had grown to ~490 KB / 8,600 lines — unreadable whole and a
> rebase magnet every parallel terminal had to touch. Fragments removed both
> problems (one file per task = nothing to collide on). The full original
> content remains in git history; the re-verification trailer summary moved to
> `docs/worklog/reverify-2026-05-26-summary.md`.

## Scenario → fragment map

| Sn | Fragment |
|---|---|
| S1 | [`docs/worklog/s1-single-snapshot-trader-session.md`](worklog/s1-single-snapshot-trader-session.md) |
| S2 | [`docs/worklog/s2-multi-day-rolling-wheel-campaign-4-weeks.md`](worklog/s2-multi-day-rolling-wheel-campaign-4-weeks.md) |
| S3 | [`docs/worklog/s3-build-wheeltracker-suggest-rolls.md`](worklog/s3-build-wheeltracker-suggest-rolls.md) |
| S4 | [`docs/worklog/s4-account-size-constrained-book-selection.md`](worklog/s4-account-size-constrained-book-selection.md) |
| S5 | [`docs/worklog/s5-live-mcp-chart-in-the-loop.md`](worklog/s5-live-mcp-chart-in-the-loop.md) |
| S6 | [`docs/worklog/s6-theta-provider-with-real-chains.md`](worklog/s6-theta-provider-with-real-chains.md) |
| S7 | [`docs/worklog/s7-advisor-committee-deep-dive.md`](worklog/s7-advisor-committee-deep-dive.md) |
| S8 | [`docs/worklog/s8-wheel-cycle-to-completion.md`](worklog/s8-wheel-cycle-to-completion.md) |
| S9 | [`docs/worklog/s9-adversarial-gate-stress.md`](worklog/s9-adversarial-gate-stress.md) |
| S10 | [`docs/worklog/s10-news-sentiment-downgrade-path.md`](worklog/s10-news-sentiment-downgrade-path.md) |
| S11 | [`docs/worklog/s11-regime-shift-stress.md`](worklog/s11-regime-shift-stress.md) |
| S12 | [`docs/worklog/s12-tradingview-webhook-ingest.md`](worklog/s12-tradingview-webhook-ingest.md) |
| S13 | [`docs/worklog/s13-dashboard-end-to-end.md`](worklog/s13-dashboard-end-to-end.md) |
| S14 | [`docs/worklog/s14-strangle-timing-gated-entry.md`](worklog/s14-strangle-timing-gated-entry.md) |
| S15 | [`docs/worklog/s15-portfolio-aggregation-gap-pro-book-level-queries.md`](worklog/s15-portfolio-aggregation-gap-pro-book-level-queries.md) |
| S16 | [`docs/worklog/s16-compliance-audit-walkthrough-single-trade-depth.md`](worklog/s16-compliance-audit-walkthrough-single-trade-depth.md) |
| S17 | [`docs/worklog/s17-week-in-the-life-operational-stress-10-trading-d.md`](worklog/s17-week-in-the-life-operational-stress-10-trading-d.md) |
| S18 | [`docs/worklog/s18-load-scale-stress-production-scale-sp500-charact.md`](worklog/s18-load-scale-stress-production-scale-sp500-charact.md) |
| S19 | [`docs/worklog/s19-failure-mode-chaos-fail-closed-contract.md`](worklog/s19-failure-mode-chaos-fail-closed-contract.md) |
| S20 | [`docs/worklog/s20-engine-api-py-concurrency-crash-resilience.md`](worklog/s20-engine-api-py-concurrency-crash-resilience.md) |
| S21 | [`docs/worklog/s21-d17-confirm-fixed-pro-account-sizing-at-1m.md`](worklog/s21-d17-confirm-fixed-pro-account-sizing-at-1m.md) |
| S22 | [`docs/worklog/s22-roll-defense-economics-itm-short-put-with-7-dte.md`](worklog/s22-roll-defense-economics-itm-short-put-with-7-dte.md) |
| S23 | [`docs/worklog/s23-earnings-window-navigation-event-gate-iv-crush-o.md`](worklog/s23-earnings-window-navigation-event-gate-iv-crush-o.md) |
| S24 | [`docs/worklog/s24-multi-strategy-book-composition-500k-wheel-cc-st.md`](worklog/s24-multi-strategy-book-composition-500k-wheel-cc-st.md) |
| S25 | [`docs/worklog/s25-vol-shock-recovery-mu-2026-03-18-earnings-beat-b.md`](worklog/s25-vol-shock-recovery-mu-2026-03-18-earnings-beat-b.md) |
| S26 | [`docs/worklog/s26-mid-cycle-re-evaluation-realism-aapl-challenged.md`](worklog/s26-mid-cycle-re-evaluation-realism-aapl-challenged.md) |
| S28 | [`docs/worklog/s28-cc-dividend-realism-vz-jpm-msft-ko-aapl-wmt.md`](worklog/s28-cc-dividend-realism-vz-jpm-msft-ko-aapl-wmt.md) |
| S29 | [`docs/worklog/s29-skew-data-absence-on-bloomberg-connector.md`](worklog/s29-skew-data-absence-on-bloomberg-connector.md) |
| S30 | [`docs/worklog/s30-hmm-regime-multiplier-realism-april-2025-vol-spi.md`](worklog/s30-hmm-regime-multiplier-realism-april-2025-vol-spi.md) |
| S31 | [`docs/worklog/s31-compounding-crisis-stress-six-concurrent-downgra.md`](worklog/s31-compounding-crisis-stress-six-concurrent-downgra.md) |
| S32 | [`docs/worklog/s32-1m-friction-modeled-simulation-closes-s22-caveat.md`](worklog/s32-1m-friction-modeled-simulation-closes-s22-caveat.md) |
| S33 | [`docs/worklog/s33-multi-stress-engine-soundness-verification.md`](worklog/s33-multi-stress-engine-soundness-verification.md) |
| S35 | [`docs/worklog/s35-2018-2020-out-of-window-cross-validation.md`](worklog/s35-2018-2020-out-of-window-cross-validation.md) |
| S36 | [`docs/worklog/s36-multi-ticker-hmm-regime-realism.md`](worklog/s36-multi-ticker-hmm-regime-realism.md) |
| S37 | [`docs/worklog/s37-ranking-philosophy-ev-dollars-vs-roc.md`](worklog/s37-ranking-philosophy-ev-dollars-vs-roc.md) |
| S38 | [`docs/worklog/s38-multi-window-backtest-at-100-tickers-1m-2020-202.md`](worklog/s38-multi-window-backtest-at-100-tickers-1m-2020-202.md) |
| S40 | [`docs/worklog/s40-rolling-multi-window-backtest-at-100-tickers-1m.md`](worklog/s40-rolling-multi-window-backtest-at-100-tickers-1m.md) |
| S41 | [`docs/worklog/s41-f4-fix-validation-backtest-post-260.md`](worklog/s41-f4-fix-validation-backtest-post-260.md) |
| S42 | [`docs/worklog/s42-r9-r10-reviewer-audit.md`](worklog/s42-r9-r10-reviewer-audit.md) |
| S43 | [`docs/worklog/s43-rolling-5-window-backtest-with-post-260-engine.md`](worklog/s43-rolling-5-window-backtest-with-post-260-engine.md) |
| S44 | [`docs/worklog/s44-s38-re-run-on-post-f4-engine-pr-260-dollar-impro.md`](worklog/s44-s38-re-run-on-post-f4-engine-pr-260-dollar-impro.md) |
| S46 | [`docs/worklog/s46-re-verify-closed-tests-on-post-260-262-engine.md`](worklog/s46-re-verify-closed-tests-on-post-260-262-engine.md) |
