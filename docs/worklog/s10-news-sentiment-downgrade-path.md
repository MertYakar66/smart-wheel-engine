---
id: S10
title: News-sentiment downgrade path
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Validate the news-sentiment overlay
(`engine/news_sentiment.py`, the only news module on the EV path):
confirm bad sentiment downgrades a candidate's EV, and — the §2
invariant — confirm good sentiment can never rescue a non-tradeable
candidate.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `as_of=2026-03-20`, 35-DTE /
25-delta. Probe A — `NewsSentimentReader` against a synthetic store in
a temp dir, mapping `sentiment_multiplier` across the sentiment range
including the extremes. Probe B — `rank_candidates_by_ev` end-to-end
with `get_ticker_sentiment` monkeypatched to inject synthetic sentiment
(process-local; the real `sentiment_multiplier` and `wheel_runner`
wiring still run), `use_news_sentiment` toggled on vs off. No code
changes.

**Path.** `use_news_sentiment` → `NewsSentimentReader.sentiment_multiplier`
→ `news_mult`, folded into `combined_regime_mult = hmm × skew × news ×
credit` (`wheel_runner.py:933`) → `trade.regime_multiplier` →
`EVEngine.evaluate`, where `ev_dollars = ev_raw × regime_mult`
(`ev_engine.py:502`). News scales the final dollar EV; it never touches
`ev_raw`.

**Status.** Done. §2 holds — bullish sentiment (capped at +5 %) cannot
rescue a negative-EV or event-gated candidate. Bad sentiment downgrades
EV as designed. No §2 violation, no bug; all findings logged.

**Findings:**

- **News multiplier is clamped to [0.88, 1.05] — asymmetric,
  downgrade-biased.** `sentiment_multiplier` is a 4-value step
  function: sentiment ≤ −0.3 → 0.88, ≤ −0.1 → 0.95, ≥ 0.3 → 1.05, else
  1.00; `n_articles < 5` forces 1.00. Probed across the full range —
  even maximal sentiment (+1.0, n=9999) caps at 1.05, minimal (−1.0)
  floors at 0.88. Max downgrade −12 %, max boost +5 %, analogous to the
  dealer multiplier's [0.70, 1.05] (CLAUDE.md §2). §2-safe by
  construction. **Logged.**

- **Bad sentiment downgrades EV — confirmed end-to-end.** News-on vs
  news-off through the ranker: CF (bearish −0.6) `ev_dollars`
  247.76 → 218.03 (×0.88); AJG (−0.2) 326.57 → 310.24 (×0.95). The
  on/off ratio equals the emitted `news_multiplier` exactly — the
  overlay works as designed. **Logged.**

- **§2 holds — good news cannot rescue.** `ev_dollars = ev_raw ×
  regime_mult` with a strictly-positive multiplier is sign-preserving:
  a negative-EV name stays negative. Probe: MSFT (negative-EV) +
  maximally-bullish sentiment (1.05) → `ev_dollars` −24.47 → −25.69
  (still < 0), and MSFT stays absent from a `min_ev_dollars=0` ranking.
  Event-gated XOM + bullish news stays gated — the event lockout
  (`ev_engine.py:262`) precedes the multiplier. Because the multiplier
  scales signed magnitude, bullish news on a negative-EV name makes
  `ev_dollars` *more* negative — harmless for §2 (such names are
  non-tradeable regardless), but it means the multiplier is a magnitude
  scaler, not a directional tilt. **Logged.**

- **`n_articles < 5` → forced neutral (1.00).** FDS with +0.90
  sentiment but only 3 articles → multiplier 1.00, zero EV change. A
  genuine sentiment signal on a thinly-covered name is silently
  ignored. **Logged.**

- **No news store on the Bloomberg setup — the overlay is a dormant
  no-op by default.** None of `news_sentiment.py`'s `_CANDIDATE_PATHS`
  (`data_processed/news_sentiment.{parquet,csv}`,
  `data/news/sentiment.*`, `financial_news/storage/sentiment.sqlite`)
  exist; with `use_news_sentiment=True` (the default) every candidate
  gets a 1.00 multiplier. Absent news — no store, no row for a ticker,
  or only stale rows — collapses silently to neutral. **Logged.**

- **News time-handling — a PIT leak, plus silent staleness.**
  `get_ticker_sentiment` keeps only rows with `as_of ≥ now() − 72h`
  using wall-clock `now()`; the ranker's `as_of` PIT cutoff never
  reaches the news reader. A backtest at `as_of=2026-03-20` would
  apply *today's* news — look-ahead, the same family as S1's `as_of`
  footgun. Conversely, news older than 72 h is dropped to neutral with
  no warning. **Fixed in #119.**

- **The overlay is surfaced — but only conditionally.** Unlike S9's
  silent ranker gates, the applied multiplier is visible:
  `news_multiplier`, `news_sentiment`, `news_n_articles` are emitted —
  but only as diagnostic columns (`include_diagnostic_fields=True`).
  With diagnostics off, the news adjustment is invisible. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `NewsSentimentReader.sentiment_multiplier(...)` returns 1.0 on missing-store path (silent neutral) by design; `wheel_runner` folds it into `combined_regime_mult` (`hmm × skew × news × credit`) so it scales `ev_dollars` only; `ev_raw` (the EV authority's input) is untouched. The "good news cannot rescue" invariant is sign-preserving by construction.
  - Qualitative verdict: match — overlay is dormant on Bloomberg setup (no news store on the worktree); `news_multiplier`, `news_sentiment`, `news_n_articles` columns all surfaced in diagnostic mode. PR #119 (news PIT-leak fix referenced in original S10 entry) shipped on `main`.
  - Numerical drift > 5%: not applicable — original entry was monkeypatched probes (CF -0.6 → ×0.88 = 247.76 → 218.03 etc.); re-running monkeypatched is out of scope this pass. The base behavior (no store → multiplier 1.0) verified directly.
  - Notes: bigger evidence that the news-PIT-leak fix shipped lives in S11's re-verify (where credit_multiplier — sister overlay — now varies with as_of date, confirming the time-handling fix landed).
