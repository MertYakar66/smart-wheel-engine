# Heavy verify HT-C — news-mult measurement + prob_profit calibration re-verify

**Card:** HT-C, cycle 2026-05-30 "heavy-verify" ([GO comment](https://github.com/MertYakar66/smart-wheel-engine/issues/113#issuecomment-4581580777)).
**Engine SHA:** `56c671d` (post-#288 cycle close; main as of 2026-05-30).
**Terminal:** C. Branch: `claude/heavy-news-calibration-reverify`.
**Driver:** `docs/verification_artifacts/news_calibration_driver.py`.
**Raw output:** `docs/verification_artifacts/heavy_news_calibration_2026-05-30_raw_output.txt`.
**Status:** §A + §B + §C complete. §C originally framed as a row-aligned `prob_profit` byte-identity check; revised post-execution (see §C "Why this was revised") to the structural pre-multiplier argument from `ev_engine.py:385` vs `:520`, with the attempted empirical rerun retained as F4/R10-confound evidence rather than as the no-op proof.

---

## TL;DR

**The news-severance "re-baseline mandatory" claim in `docs/NEWS_REDESIGN_CAMPAIGN.md` §3 is FALSE on this Bloomberg-only environment.** Empirically (§A), the pre-D18 `sentiment_multiplier` returned **1.0 in 1300 of 1300 probes** spanning the full S20-S40 measurement window, because the news_sentiment store has never been populated. D18 changes the value of `news_mult` from 1.0 to 1.0 — a true no-op for the EV path's regime multiplier composition. Structurally (analytical, §C), `prob_profit` is invariant to `news_mult` by construction: `prob_profit = float(np.mean(pnls > 0))` at `engine/ev_engine.py:385` runs strictly **before** the multiplier is applied at `engine/ev_engine.py:520` (`ev_dollars = ev_raw * regime_mult`). Changing `news_mult` (D18's only effect) cannot move `prob_profit`. The 10-config calibration measurement in §B reproduces every published value from `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` byte-identically (under the same `otm_expire` exit-attribution methodology the published doc used — see §B verdict footnote on HT-B reconciliation) — the −15 to −18pp top-bin over-confidence on $1M / 100t configs (S34, S35, S38, S38-postF4, S40-W1/W2/W3) is reproduced exactly.

**Escalation verdict:** **NOT escalating to a full S34-class re-run.** The card escalation clause was "ONLY IF the measured news_mult is non-1.0"; §A shows it was 1.0 always. §C originally framed an attempted empirical confirmation via a row-aligned `prob_profit` byte-equality on a post-#249 S27 rerun. That framing was revised after Session C's review: the rerun would have routed through the **current** engine (post-#260 F4 widening + post-#262 R10 cap + post-#249 D18), so any diff against the pre-F4 on-disk `rank_log.csv` would conflate D18 with F4 and R10 (the actual rerun also failed the row-aligned join because the post-#249 rank_log schema dropped the `exit_reason` column — `KeyError` at driver L226). The no-op proof rests on §A + the `ev_engine.py:385` vs `:520` structural argument; the attempted §C empirical confirmation is now documented as F4/R10-confound evidence rather than as the proof itself. A full $1M / 100t / 3-year re-baseline would cost ~3.5h of compute to confirm a structurally proven invariant.

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

The prob_profit calibration finding is robust **under the same `otm_expire` exit-attribution methodology the published doc used** (see HT-B reconciliation footnote below):

1. **Top bin (0.95, 1.0] is MISCAL by −15 to −18pp on every $1M / 100-ticker config** (S34, S35, S38, S38-postF4, S40-W1/W2/W3). 7 of 7 large-scale configs show the structural defect.
2. **Smaller-scale configs ($100k / 24-ticker) show milder top-bin miscalibration** (S22 −7.75pp, S27 −4.88pp, S32 −5.11pp) — but the *low* bin `(0.5, 0.6]` shows the symmetric opposite (engine UNDER-claims by +20pp on S27 → 9 / 10 configs in the published doc). This is consistent with the structural mechanism documented in PR #273: empirical forward distributions cannot represent "unseen tail events that haven't happened in the sample but could happen."
3. **F4 fix does NOT improve calibration**: S38 pre/post-F4 give 6.45 → 6.40pp MAD (essentially identical, slight noise) and top-bin −14.80 → −15.63pp (slight worsening, within noise). This reproduces the published finding exactly.

**HT-B (#290) reconciliation footnote.** The card HT-B explored an alternative
exit-attribution methodology — *engine-EXACT* P&L attribution, which traces
the realized outcome through the same probabilistic forward distribution the
engine used to score the candidate — and found the top-bin (0.95, 1.0]
miscalibration is **regime-dependent**: the in-sample S38 top bin flips from
"MISCAL −14.80pp" (under `otm_expire`) to "OK" (under engine-EXACT) — a
~12pp methodology artifact, not contradictory to §B but worth surfacing here
so the two cards reconcile. §B's "robust" verdict therefore reads:
*robust under the published `otm_expire` methodology the PROB_PROFIT_CALIBRATION
doc used; HT-B (#290) shows the same finding is regime-dependent under
engine-EXACT attribution.* Both are legitimate framings of the same underlying
question; the operator should be aware that the "−15 to −18pp" figure is
methodology-conditional, not a fact about the engine in isolation.

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

## §C — `prob_profit` invariance under D18 (structural proof + F4/R10-confound evidence)

**Status:** revised. The original framing (row-aligned `prob_profit` byte-equality
between the existing pre-#249 `s27_backtest/rank_log.csv` and a post-#249 S27 rerun)
was found, on Session C's verification review of the captured run, to be
methodologically unsound: the rerun routes through the **current** engine
(post-#249 D18 **plus** post-#260 F4 widening **plus** post-#262 R10 cap), so any
non-zero `prob_profit` diff would be F4-attributable, not D18-attributable. The
section is reframed below around the tighter argument: a code-level structural
proof at `engine/ev_engine.py:385` vs `:520`, with the attempted empirical rerun
retained as F4/R10-confound evidence rather than as the no-op proof.

### The structural proof (this is the no-op proof for D18)

**`prob_profit` is invariant to `news_mult` by construction.**

`engine/ev_engine.py:385` (post-#249 main, SHA `75eda2e`):

```python
prob_profit = float(np.mean(pnls > 0))
```

`engine/ev_engine.py:520` (same SHA):

```python
ev_dollars = ev_raw * regime_mult
```

`news_mult` enters the EV path only through
`combined_regime_mult = hmm × skew × news × credit` at
`engine/wheel_runner.py:~1461`, which is passed in as `trade.regime_multiplier`
and consumed exclusively at `ev_engine.py:520`. **Line 385 runs strictly before
line 520, on a `pnls` array `regime_mult` never touched.** Therefore changing
`news_mult` (D18's only effect — it makes `sentiment_multiplier()` always
return `1.0`) cannot move `prob_profit`. Pinned in the test suite by
`tests/test_ev_engine_percentiles.py::TestEVResultPercentiles::test_percentiles_are_pre_multiplier`,
which exercises the same `pnls` array → percentile-field code path on which
`prob_profit` also rides.

Combined with §A (D18's effect on `news_mult` was empirically 0/1300 cells on
Bloomberg), this is the airtight no-op proof: D18 cannot move `prob_profit`
**by code structure**, and it does not change `news_mult` either **by
measurement**. The published `PROB_PROFIT_CALIBRATION_2026-05-28.md` numbers
remain the post-#249 engine baseline without any rerun.

### What the attempted empirical rerun showed (F4/R10-confound evidence)

The original §C plan executed the `s27_ivpit_24t_100k` harness against the
current engine SHA `75eda2e` (post-#249 + post-#260 F4 + post-#262 R10) and
attempted to join the new rank_log to the pre-F4 on-disk rank_log under
`%TEMP%\s27_backtest\rank_log.csv`. Captured aggregate metrics:

| Metric | New rerun (post-everything) | Published S27 baseline (pre-#260 / pre-#262 / pre-#249) |
|---|---:|---:|
| `row_count` | 5,944 | 5,944 ✓ (entry-day candidate count is harness-deterministic) |
| `spearman_rho` (EV vs realized P&L) | **+0.1819** | **+0.188** (memory note: pre-F4 baseline) |
| `executed_trades` (rank → tracker pipeline) | 40 | harness-dependent; the W3 / S38 memory note documents ~3-10× variance between `_common.py` and throwaway harnesses on the same engine + window |
| `hit_rate` | 0.804 | — |
| `ev_mean` | −$21.07 | — |
| `mean_realized` | +$50.79 | — |

The row-aligned `prob_profit` byte-comparison **was not computable**:
the post-#249 / post-F4 rank_log no longer carries the `exit_reason` column the
join script keyed on (`KeyError: 'exit_reason'` at driver line 226). The
column drift is itself a downstream artifact of the post-#260 / post-#262
rank-log schema evolution since the pre-F4 baseline; it mechanically prevents
the originally-framed row-aligned comparison.

**Why the deltas (Spearman ρ +0.188 → +0.182, executed_trades shift) are not D18:**

| Change | Effect on the diff |
|---|---|
| **D18 (#249)** — `news_mult` stub to 1.0 | **0** in 1300/1300 cells (§A); cannot move `prob_profit` structurally (§C above) |
| **F4 widening (#260)** — RV30/RV252 multiplier on `forward_log_returns` | Changes the input distribution to `EVEngine.evaluate` → changes `pnls` → genuinely moves `prob_profit` in vol-elevated regimes. Memory note `f4-widening-overfires-on-hmm-labels` records this same S27 ρ +0.188 → +0.182 shift on a different F4 variant; the realized-vol-ratio variant that shipped preserves the ρ direction but shifts the magnitude slightly |
| **R10 single-name cap (#262)** — refuses candidates whose first-contract notional > 10% × NAV | Changes the **executed-trade set** on the rank → roll pipeline (drops AZO + BKNG entirely on S34 per memory note), not the rank_log rows themselves. Explains the `executed_trades` divergence, not the ρ |
| Harness execution-selection delta | Memory note `harness-vs-throwaway-execution-delta` records that the same engine + window can produce W3 = 516 vs S38 = 305 executed puts depending on the harness driver. Contributes to the `executed_trades=40` figure |

The Session C call-out is exactly right: **the row-aligned comparison conflates
D18 + F4 + R10 + harness, not D18 alone.** To isolate D18 empirically would
require running `edba283` (PR #249's parent) vs `9e8edcd` (PR #249's merge)
at identical engine SHA otherwise — but that is structurally what
`ev_engine.py:385` vs `:520` already proves analytically without a 25-minute
compute.

### §C Verdict

**`prob_profit` is structurally invariant under D18.** The no-op proof for the
"NEWS_REDESIGN_CAMPAIGN re-baseline mandatory" claim rests on
§A (empirical) + the `ev_engine.py:385` vs `:520` structural argument above.
The attempted row-aligned byte-equality test is **dropped as methodologically
unsound** — it would have measured F4 + R10 + harness, not D18 alone, and the
post-#249 rank_log schema drift mechanically prevents the join anyway. The
captured aggregate metrics are retained in the raw output as evidence of the
F4 + R10 + harness deltas, not as D18 evidence.

The "≤ 1e-6 max_abs_diff" acceptance gate from the original framing is
**revoked**. The new acceptance gate is the structural argument: line 385 runs
strictly before line 520 on a `pnls` array the multiplier never touches;
`news_mult` enters only through `regime_mult` at line 520; therefore D18
cannot move `prob_profit` by construction. **Independently verified by
Session C against the same lines on main** — Session C concurred: "D18 (which
only changes news_mult → only scales ev_dollars) cannot move prob_profit →
calibration is structurally invariant. C's 'line 385' citation is exact. ✓
… The 'D18 = no-op' conclusion is airtight."

---

## Findings (sharp, for the Major Session to triage next cycle)

### F1 — `NEWS_REDESIGN_CAMPAIGN.md` §3 "re-baseline mandatory" claim needs scope tightening

The "every prior `S<N>` scenario result becomes obsolete after this campaign" framing in `docs/NEWS_REDESIGN_CAMPAIGN.md` §3 (lines 122-128) conflates four orthogonal changes (D18 + R9 + EDGAR + FRED). For D18 alone on this Bloomberg-only environment, the claim is **measurably false** — the §A measurement shows news_mult was 1.0 in 1300/1300 cells. The doc's framing should narrow to "re-baseline mandatory **after PR 5 (R9) ships** (introduces a NEW reviewer rule with non-trivial firing rate)" or equivalent. **Not a bug in the engine** — a doc-currency issue surfaced by this card.

Suggested edit (next cycle, magnet-doc owner): replace §3's "Backtest re-baseline is mandatory" subsection with:

> Re-baseline is mandatory after R9 (PR 5) and the EDGAR event-gate replacement (PR 3.5) ship. D18 alone is a no-op on the Bloomberg-only environment (`docs/HEAVY_NEWS_CALIBRATION_REVERIFY.md` §A: news_mult = 1.0 in 1300/1300 probes); the FRED rewrite (PR 6) substitutes one signal for another. Until R9 and EDGAR-gate change the candidate set, the published `S<N>` numbers remain the engine baseline.

### F2 — News pipeline outputs nothing operationally

`scripts/pull_news_sentiment.py` exists and writes to `data_processed/news_sentiment.parquet`. The output file has never been on disk in git history. No `POLYGON_API_KEY` is wired to CI or the dev-box bootstrap. The downstream dashboard column reads from the same empty store and shows a no-op column. **Not a regression** (D18 makes this OK on the EV side), but a transparency gap on the operator side: the dashboard surfaces a `news_sentiment` column that is always 0.0 / N=0. Either retire the puller (operator decision, per the D18 commit's "Unresolved" item) or document the column's zero-state in the dashboard.

### F3 — A complete post-#260 multi-config calibration re-baseline is outside HT-C scope

`PROB_PROFIT_CALIBRATION_2026-05-28.md` was authored against the rank_logs §B
re-derived from. The post-#260 engine has shifted slightly (F4 RV widening
adds tail mass in vol-elevated regimes), but the rank_logs the doc cites are
pre-#260 for S22-S38 and post-#260 only for S38-postF4. §C's attempted S27
rerun was originally framed as "one post-#249 + post-#260 calibration data
point"; the revision (see §C) drops that framing because the same rerun is
F4/R10-confounded for D18 isolation. A complete post-#260 multi-config
re-baseline of S34 / S38 / S40 is outside HT-C's measure-first scope
(would require ~7h of compute) and is the open item
`PROB_PROFIT_CALIBRATION_2026-05-28.md` §"For future research" already
names (POT-GPD wired into prob_profit). HT-B (#290)'s engine-EXACT
attribution methodology is a related, complementary line.

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

- **§C revision rationale (closed)**: the original row-aligned `prob_profit`
  byte-equality framing was revised post-Session-C review to a structural
  argument (`ev_engine.py:385` vs `:520`) — the attempted rerun is preserved
  in the raw output as F4/R10-confound evidence, not as the no-op proof. The
  no-op proof is §A + the structural argument; both pillars are tight.
- **F1 (doc-currency)**: surfaces to the magnet-doc reconciliation at the
  next cycle close. The suggested NEWS_REDESIGN_CAMPAIGN §3 edit is in F1
  above verbatim.
- **F3 (post-#260 multi-config re-baseline)**: not in HT-C scope; a
  follow-on card if the operator wants a complete post-#260 calibration
  matrix beyond S38-postF4. Pairs naturally with HT-B (#290)'s engine-EXACT
  attribution methodology.
- **HT-B (#290) reconciliation**: noted in §B verdict footnote.
  PROB_PROFIT_CALIBRATION_2026-05-28.md's "−15 to −18pp" figure is
  `otm_expire`-methodology-conditional and flips under engine-EXACT
  attribution; both framings are legitimate and answer slightly different
  questions. No correction to either card; just the qualifier so a future
  agent doesn't read the two findings as contradictory.
