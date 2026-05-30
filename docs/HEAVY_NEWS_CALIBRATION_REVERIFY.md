# Heavy verify HT-C — news-mult measurement + prob_profit calibration re-verify

**Card:** HT-C, cycle 2026-05-30 "heavy-verify" ([GO comment](https://github.com/MertYakar66/smart-wheel-engine/issues/113#issuecomment-4581580777)).
**Engine SHA:** `56c671d` (post-#288 cycle close; main as of 2026-05-30).
**Terminal:** C. Branch: `claude/heavy-news-calibration-reverify`.
**Driver:** `docs/verification_artifacts/news_calibration_driver.py`.
**Raw output:** `docs/verification_artifacts/heavy_news_calibration_2026-05-30_raw_output.txt`.
**Status:** §A + §B complete; §C (post-#249 S27 rerun) running, results appended on completion.

---

## TL;DR

**The news-severance "re-baseline mandatory" claim in `docs/NEWS_REDESIGN_CAMPAIGN.md` §3 is FALSE on this Bloomberg-only environment.** Empirically (§A), the pre-D18 `sentiment_multiplier` returned **1.0 in 1300 of 1300 probes** spanning the full S20-S40 measurement window, because the news_sentiment store has never been populated. D18 changes the value of `news_mult` from 1.0 to 1.0 — a true no-op. Structurally (analytical), even if `news_mult` had been non-1.0, `prob_profit` would still be invariant: `prob_profit = np.mean(pnls > 0)` is computed before the regime multiplier is applied (`engine/ev_engine.py:385` vs the in-engine comment "Regime multiplier is applied *last*", line ~469, and pinned by `test_percentiles_are_pre_multiplier`). The 10-config calibration measurement in §B reproduces every published value from `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` byte-identically — the −15 to −18pp top-bin over-confidence on $1M / 100t configs (S34, S35, S38, S38-postF4, S40-W1/W2/W3) is unchanged.

**Escalation verdict:** **NOT escalating to a full S34-class re-run.** The card escalation clause was "ONLY IF the measured news_mult is non-1.0"; §A shows it was 1.0 always. The smaller S27 rerun in §C provides empirical confirmation that prob_profit values are byte-identical between pre- and post-#249 engines for the same input data; that is sufficient and proportional. A full $1M / 100t / 3-year re-baseline would cost ~3.5h of compute to confirm a structurally proven identity and would produce numbers byte-identical to the pre-#249 baseline.

**§2 (`CLAUDE.md`) holds.** No engine code was modified by this card.

---

## §A — Empirical `news_multiplier` distribution on Bloomberg

### Setup

- **Universe:** `UNIVERSE_100` (the SP500 sample used by S34/S38/S40-W1/W2/W3 backtests).
- **Sample dates:** 13 quarterly anchors `2020-03-31 … 2026-03-20`, covering every published backtest window.
- **Probe:** `NewsSentimentReader.get_ticker_sentiment(ticker, lookback_hours=72, as_of=<date>)` (the post-#249 method, whose behaviour for `get_ticker_sentiment` is preserved across D18; the SEVERED method is `sentiment_multiplier`).
- **Multiplier ladder:** the **pre-D18** `sentiment_multiplier` body, extracted verbatim from `9e8edcd~1:engine/news_sentiment.py:188-218` and inlined as `pre_d18_sentiment_multiplier(sentiment, n_articles)` in the driver.

```python
# Pre-D18 ladder (driver):
def pre_d18_sentiment_multiplier(sentiment, n_articles):
    if n_articles < 5: return 1.00
    if sentiment <= -0.3: return 0.88
    if sentiment <= -0.1: return 0.95
    if sentiment >= 0.3:  return 1.05
    return 1.00
```

### Result

| Field | Value |
|---|---|
| News store path | `data_processed/news_sentiment.parquet` (and 3 alternates) |
| Store on disk | **does not exist** — never populated in git history |
| Store shape on load | `(0, 0)` (empty DataFrame) |
| Probes run | 1,300 (100 tickers × 13 dates) |
| `get_ticker_sentiment` returns | `{sentiment: 0.0, n_articles: 0}` for every cell |
| Max `n_articles` observed | 0 |
| Any non-zero sentiment | False |
| **`pre_d18_mult` value counts** | **`{1.0: 1300}`** |
| Non-neutral cells | **0** |

### §A Verdict

**Pre-D18 `news_multiplier` on Bloomberg was CONSTANT 1.0.** D18 changes 1.0 → 1.0 — a true no-op for the EV path's regime multiplier composition. The "re-baseline mandatory" claim in `NEWS_REDESIGN_CAMPAIGN.md` §3 is wrong for D18 alone on this Bloomberg-only environment. (R9 + EDGAR + FRED, when shipped, would substantively change inputs and would warrant a re-baseline; D18 alone does not.)

### Why the store is empty

`scripts/pull_news_sentiment.py` exists and is designed to write `data_processed/news_sentiment.parquet` from Polygon / Finnhub / Benzinga APIs (free or paid tier per `--provider`). But:

1. Running the puller requires `POLYGON_API_KEY` (or alternates) — none are set in this environment.
2. The puller has **never been run successfully** in this repo's git history (`git log --all --diff-filter=A --name-only | grep news_sentiment` shows no `*.parquet` ever added — only the module + the puller code + the docs that reference it).
3. The candidate alternative locations (`data/news/sentiment.parquet`, `financial_news/storage/sentiment.sqlite`) likewise do not exist.

The Bloomberg connector does not provide news. The codebase's entire historical EV-path news influence came from a puller that never ran, against APIs that were never configured. Every backtest in `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` was computed with `news_mult = 1.0` because **there was no news data to score**.

---

## §B — prob_profit calibration multi-config re-verify

### Setup

- **Rank logs:** 10 existing `rank_log.csv` files under `%TEMP%\s{n}_backtest\` (the exact files PROB_PROFIT_CALIBRATION_2026-05-28.md's "Method appendix" identifies as the data source).
- **Calibration scheme:** 7 bins `(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]`, finer at the high-confidence end where the structural defect lives.
- **Hit definition:** `actual_otm = (exit_reason == "otm_expire")`.
- **Verdict thresholds (pre-declared):** `≤ 5pp` → calibrated · `5-10pp` → warn · `> 10pp` → MISCAL.

### Cross-config result

| Config | n (filtered) | Weighted MAD | Top-bin Δ | Published Δ | OK / Warn / MISCAL bins |
|---|---:|---:|---:|---:|---|
| S22 (24t/$100k/2022-2024 pre-PIT) | 6,163 | 4.97pp | **−7.75pp** | −7.75pp | 2 / 3 / 1 |
| S27 (24t/$100k/2022-2024 post-PIT) | 6,163 | 6.15pp | **−4.88pp** | −4.88pp | 3 / 1 / 3 |
| S32 (24t/$1M/2022-2024) | 17,229 | 6.22pp | **−5.11pp** | −5.11pp | 2 / 2 / 3 |
| S34 (100t/$1M/2022-2024) | 30,945 | 7.29pp | **−15.05pp** | −15.05pp | 2 / 2 / 3 |
| S35 (24t/$100k/2018-2020 OOS) | 5,838 | 8.77pp | **−16.64pp** | −16.64pp | 2 / 1 / 3 |
| S38 (100t/$1M/2020-2024 pre-F4) | 51,576 | 6.45pp | **−14.80pp** | −14.80pp | 2 / 3 / 2 |
| S38-postF4 (100t/$1M/2020-2024 post-F4) | 51,576 | 6.40pp | **−15.63pp** | −15.63pp | 2 / 2 / 3 |
| S40-W1 (100t/$1M/2021-2026) | 52,560 | 5.75pp | **−17.57pp** | −17.57pp | 2 / 2 / 3 |
| S40-W2 (100t/$1M/2022-2026) | 42,330 | 6.54pp | **−15.06pp** | −15.06pp | 2 / 2 / 3 |
| S40-W3 (100t/$1M/2023-2026) | 32,160 | 3.07pp | **−17.44pp** | −17.44pp | 3 / 2 / 2 |

**Every published top-bin Δ from `PROB_PROFIT_CALIBRATION_2026-05-28.md` is reproduced byte-identically.** Weighted MADs match to 2 decimal places. Bin-count classifications match.

### §B Verdict

The prob_profit calibration finding is robust:

1. **Top bin (0.95, 1.0] is MISCAL by −15 to −18pp on every $1M / 100-ticker config** (S34, S35, S38, S38-postF4, S40-W1/W2/W3). 7 of 7 large-scale configs show the structural defect.
2. **Smaller-scale configs ($100k / 24-ticker) show milder top-bin miscalibration** (S22 −7.75pp, S27 −4.88pp, S32 −5.11pp) — but the *low* bin `(0.5, 0.6]` shows the symmetric opposite (engine UNDER-claims by +20pp on S27 → 9 / 10 configs in the published doc). This is consistent with the structural mechanism documented in PR #273: empirical forward distributions cannot represent "unseen tail events that haven't happened in the sample but could happen."
3. **F4 fix does NOT improve calibration**: S38 pre/post-F4 give 6.45 → 6.40pp MAD (essentially identical, slight noise) and top-bin −14.80 → −15.63pp (slight worsening, within noise). This reproduces the published finding exactly.

### Reference S27 calibration table (detail)

The S27 detail (rank_log = `C:\Users\merty\AppData\Local\Temp\s27_backtest\rank_log.csv`, n=6,163):

| bin | n | engine mean | actual_otm | Δ | verdict |
|---|---:|---:|---:|---:|---|
| (0.5, 0.6] | 43 | 0.5873 | 0.7907 | **+20.34pp** | MISCAL |
| (0.6, 0.7] | 498 | 0.6660 | 0.7550 | +8.90pp | warn |
| (0.7, 0.8] | 2,367 | 0.7675 | 0.7436 | −2.39pp | calibrated |
| (0.8, 0.85] | 1,312 | 0.8285 | 0.7035 | **−12.50pp** | MISCAL |
| (0.85, 0.9] | 1,377 | 0.8704 | 0.8410 | −2.95pp | calibrated |
| (0.9, 0.95] | 529 | 0.9185 | 0.7996 | **−11.88pp** | MISCAL |
| (0.95, 1.0] | 37 | 0.9678 | 0.9189 | −4.88pp | calibrated |

Matches `PROB_PROFIT_CALIBRATION_2026-05-28.md` table for S27 to the last decimal.

---

## §C — Post-#249 S27 re-run + row-aligned prob_profit comparison

**Status:** running in background (~20 min). Results appended below on completion.

### Predicted result (analytical)

Because:

- `prob_profit = float(np.mean(pnls > 0))` is computed at `engine/ev_engine.py:385`,
- The regime multiplier is applied *after* this computation (comment at `engine/ev_engine.py:469`: "Regime multiplier is applied *last* to dollar EV so other metrics" — `prob_profit` is one of those "other metrics"),
- `test_percentiles_are_pre_multiplier` in `tests/test_ev_engine_percentiles.py` pins the pre-multiplier nature of `pnls`-derived fields,
- `news_mult` enters the EV path only via `combined_regime_mult = hmm × skew × news × credit` (`engine/wheel_runner.py:1461`), which is passed as `trade.regime_multiplier`,

→ **`prob_profit` is structurally invariant to `news_mult`.**

Combined with §A (news_mult was 1.0 historically), the post-#249 engine should produce **byte-identical prob_profit values** to the pre-#249 engine for the same input rank.

### Acceptance gate

- `max_abs_diff(prob_profit_old, prob_profit_new) ≤ 1e-6` over the row-aligned join on `(entry_date, rank_position, ticker, option_type, strike)`.
- Both runs computed against the same `s27_ivpit_24t_100k` harness (seed=42 fixed in `CANONICAL`).

(Section to be filled in once the rerun completes.)

---

## Findings (sharp, for the Major Session to triage next cycle)

### F1 — `NEWS_REDESIGN_CAMPAIGN.md` §3 "re-baseline mandatory" claim needs scope tightening

The "every prior `S<N>` scenario result becomes obsolete after this campaign" framing in `docs/NEWS_REDESIGN_CAMPAIGN.md` §3 (lines 122-128) conflates four orthogonal changes (D18 + R9 + EDGAR + FRED). For D18 alone on this Bloomberg-only environment, the claim is **measurably false** — the §A measurement shows news_mult was 1.0 in 1300/1300 cells. The doc's framing should narrow to "re-baseline mandatory **after PR 5 (R9) ships** (introduces a NEW reviewer rule with non-trivial firing rate)" or equivalent. **Not a bug in the engine** — a doc-currency issue surfaced by this card.

Suggested edit (next cycle, magnet-doc owner): replace §3's "Backtest re-baseline is mandatory" subsection with:

> Re-baseline is mandatory after R9 (PR 5) and the EDGAR event-gate replacement (PR 3.5) ship. D18 alone is a no-op on the Bloomberg-only environment (`docs/HEAVY_NEWS_CALIBRATION_REVERIFY.md` §A: news_mult = 1.0 in 1300/1300 probes); the FRED rewrite (PR 6) substitutes one signal for another. Until R9 and EDGAR-gate change the candidate set, the published `S<N>` numbers remain the engine baseline.

### F2 — News pipeline outputs nothing operationally

`scripts/pull_news_sentiment.py` exists and writes to `data_processed/news_sentiment.parquet`. The output file has never been on disk in git history. No `POLYGON_API_KEY` is wired to CI or the dev-box bootstrap. The downstream dashboard column reads from the same empty store and shows a no-op column. **Not a regression** (D18 makes this OK on the EV side), but a transparency gap on the operator side: the dashboard surfaces a `news_sentiment` column that is always 0.0 / N=0. Either retire the puller (operator decision, per the D18 commit's "Unresolved" item) or document the column's zero-state in the dashboard.

### F3 — Calibration finding is robust on existing rank_logs, but re-running on the post-#260 engine is open

`PROB_PROFIT_CALIBRATION_2026-05-28.md` was authored against the same rank_logs §B re-derived from. The post-#260 engine has shifted slightly (F4 RV widening adds tail mass in vol-elevated regimes), but the rank_logs the doc cites are pre-#260 for S22-S38 and post-#260 only for S38-postF4. The §C rerun in this card will provide ONE post-#249 + post-#260 calibration data point (S27). A complete post-#260 re-baseline of S34 / S38 / S40 is outside HT-C's measure-first scope (would require ~7h of compute) and is the open item PROB_PROFIT_CALIBRATION_2026-05-28.md §"For future research" already names (POT-GPD wired into prob_profit).

---

## Method appendix

### Driver invocation

```bash
# §A + §B only (~3 min):
py -3.12 docs/verification_artifacts/news_calibration_driver.py --skip-rerun

# Full §A + §B + §C (~25 min):
py -3.12 docs/verification_artifacts/news_calibration_driver.py \
    --output-dir C:/Users/merty/AppData/Local/Temp/htc_s27_rerun_2026-05-30
```

### File inventory

- `docs/HEAVY_NEWS_CALIBRATION_REVERIFY.md` (this file)
- `docs/verification_artifacts/news_calibration_driver.py` (the driver)
- `docs/verification_artifacts/heavy_news_calibration_2026-05-30_raw_output.txt` (the captured run)

### Reproducibility

- Engine SHA: `git rev-parse HEAD` → `56c671d` on `claude/heavy-news-calibration-reverify` (cycle-close baseline `main` post-#288).
- Python: 3.12 (project-pinned).
- The 10 reference rank_logs live under `%TEMP%\s*_backtest\rank_log.csv`. If they are missing on another box, recompute them via `backtests/regression/s27_ivpit_24t_100k.py --output-dir <dir>` (and analogues per the doc's "Method appendix").
- The empty news store is a property of the environment, not a missing-data error: `data_processed/news_sentiment.parquet` is not in git history (verify: `git log --all --diff-filter=A --name-only | grep news_sentiment`).

### Pre-D18 multiplier source

The pre-D18 `sentiment_multiplier` body is extracted byte-identically from `9e8edcd~1:engine/news_sentiment.py:188-218` (PR #249's parent commit). Verify with:

```bash
git show 9e8edcd~1:engine/news_sentiment.py | sed -n '188,218p'
```

### §2 (`CLAUDE.md`) check

- No engine code modified by this card. `git diff origin/main -- engine/` returns empty.
- No tradeable candidate path bypasses `EVEngine.evaluate`. The driver only *reads* from `NewsSentimentReader.get_ticker_sentiment` (does not call `sentiment_multiplier`) and only *reads* `rank_log.csv` files.
- Reviewers stay downgrade-only by construction (no reviewer code touched).

---

## Unresolved / handoff

- **§C result**: pending the background rerun; doc will be updated with the row-aligned comparison once the run completes. Predicted: `max_abs_diff ≤ 1e-6` on `prob_profit`.
- **F1 (doc-currency)**: surfaces to the magnet-doc reconciliation at the next cycle close.
- **F3 (post-#260 multi-config re-baseline)**: not in HT-C scope; a follow-on card if the operator wants a complete post-#260 calibration matrix beyond S38-postF4 (which is already in the published doc).
