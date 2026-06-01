# Theta option-history larder — scope & caveats (2026-06-01)

What the `scripts/pull_theta_option_history.py` backfill pulls, why it's scoped
this way, and the **survivor-bias caveat** any consumer must respect. The data
lands under `data_processed/theta/option_history/` (gitignored).

## Scope of the current run

| Axis | Choice | Why |
|---|---|---|
| **Universe** | Top **150** S&P names by avg dollar-turnover 2018→now (`logs/liquid_rank_turnover_2018.csv`) | The names we'd actually wheel; the long tail's bid-ask eats the premium edge. |
| **History** | **2018 → now** | Covers Volmageddon, Q4-2018, COVID, 2022 bear — every regime the forward-distribution / tail models need. |
| **Strikes** | **All** (free via the bulk per-expiration EOD call) | Full wing is future-proofing for Nelson-Siegel skew + dealer-GEX/wall; costs no extra requests. |
| **Lookback** | **90 days** per contract (`--lookback-days 90`) | The skew/GEX fit on any date D is a *cross-section* of expirations live on D; a 90-day window captures the full 0–90 DTE band on every date — the entire region a 30–45 DTE wheel + near-term skew read, plus ~30 trading days pre-entry lead-in. 210 is far-dated tail nobody reads; resumable, so extendable later. |
| **SPY/QQQ** | **Reference-only**, separate dir, `--cadence weekly` (drops Mon–Thu 0DTE), 90-day | Canonical index skew / GEX / gamma-flip surface for a future dealer-positioning signal. **Never enters the candidate ranker** (§3: no non-S&P-500 *trading*; reference data is fine). |

## ⚠️ Survivor-bias caveat — READ BEFORE BACKTESTING

The universe is ranked on the **current** 503 constituents' turnover. It therefore
**excludes names that were liquid then but have since left the index** (delisted,
acquired, or dropped). **Any backtest off this larder is survivor-biased** — and
for a short-put strategy that bias is exactly the wrong direction: it hides the
assignment disasters in names that blew up and exited (the tail we most care about).

This is a **known, documented limitation, not a silent one.** It is *not* faked
around: the `get_universe_as_of(pit_date)` PIT-membership helper
(`data/consolidated_loader.py`) can enumerate historical members, but the
current-503 turnover file cannot rank the *liquidity* of delisted names, so a
faithful fix needs a point-in-time S&P membership + historical-liquidity source.

**Follow-on (deferred, bounded):** when a PIT membership table is sourced, backfill
the once-large delisted members over 2018→now as a separate pull. Until then,
treat backtest results as survivor-biased (optimistic on tail risk).

## Other deferred (resumable, non-destructive) extensions
- **2016–2017** history — benign melt-up years, most survivorship-distorted; low value, pull on request.
- **Lookback > 90 days** — if a feature later proves it needs deeper per-contract history.
- **Universe > top-150** — the 150–180 band is a trivial resumable add.
- **Per-contract IV / Greeks-1st time series, and tick Trade/Trade-Quote** — separate later phases.

## Correctness notes (pinned by `tests/test_option_history_puller.py`)
- **History floor:** `option/history/eod` returns EMPTY for the whole range if
  `start_date` precedes the 2016-01-01 tier floor; the puller clamps to it
  (`_THETA_HISTORY_FLOOR`). Found because early-2016 pulls silently returned zero rows.
- **Atomic writes:** partitions write to `.tmp` then rename, so a crash mid-write
  cannot leave a partial file that the file-exists resume check skips as "done".
