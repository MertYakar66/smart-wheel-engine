---
id: validation-tier2-amnesia
title: Tier-2 amnesia triage — extend the truncation A/B to dated tier-2 files
kind: verification
status: completed
terminal: in-sandbox (brain)
pr:
decisions: []
date: 2026-07-14
headline: The one non-blocking validation-phase leftover, closed. V2-a proved the tier-1 market series PIT-clean but copied every tier-2 source intact on both A/B sides, so it could not see a tier-2 leak. This triage (plan §11, pre-registered before code at fb755c6) traced every tier-2 source against the short-put rank path and extended V2-a's physical-truncation A/B to the two dated tier-2 files (corporate_actions@announcement_date, dividends@declared_date). Result TIER2_PIT_CLEAN — byte-identical rank at all 5 dates with 6,000+ post-T rows dropped. Earnings is DATE-only (no outcome leak); two dormant snapshot/parquet fallbacks recorded as standing PIT limitations. No engine change ships.
surface: [backtests/freeze_replay.py, scripts/run_freeze_replay.py, tests/test_freeze_replay.py, docs/VALIDATION_PHASE_PLAN.md, docs/VALIDATION_PHASE_FINDINGS_2026-07-13.md]
---

## Goal

Resolve the §6 "tier-2 amnesia scope" caveat: V2-a's point-in-time proof
truncated only the tier-1 market series (`ohlcv, vol_iv, treasury, vix,
liquidity`) and copied every tier-2 source intact on BOTH A/B sides —
which cannot detect a tier-2 leak. Triage every tier-2 source against
the short-put rank path and, where a source is dated and loaded, extend
the physical-truncation A/B to it.

## What we tried

- **Code trace first, truncation second.** `rank_candidates_by_ev` ends
  in `EVEngine.evaluate`, a pure function — a source can only move a rank
  through a `ShortOptionTrade` field, the `EventGate`, or the regime
  multiplier. Traced each of the five loadable tier-2 keys
  (`dividends, earnings, fundamentals, credit_risk, corporate_actions`)
  to classify it, then used physical truncation as the black-box check
  the static trace cannot self-certify (V2-a's own rationale).
- Rejected truncating `earnings`: the event-lockout legitimately needs
  the next *announced* earnings DATE after `as_of` (forward-by-design);
  cutting it would manufacture a false leak. Audited instead.
- Rejected truncating the dateless snapshots (`fundamentals`,
  `credit_risk`): no date column exists to cut — inherent PIT limitation,
  same class as V2-a's credit_risk note.

## What worked

- Classification (connector `_FILES` map + full trace): 11 tier-2 CSVs
  are OFF the rank path (0 engine refs, incl. `dividends` for the SHORT-PUT
  ranker — it feeds only `analyze_ticker` + the covered-call ranker);
  `corporate_actions` is ON the path but internally gated
  `announcement_date <= as_of`; `earnings` is forward-by-design and
  DATE-only; `fundamentals`/`credit_risk` are dateless snapshots.
- `freeze_replay.py`: a `tier` parameter on `build_truncated_data_dir` /
  `amnesia_report` (tier=1 byte-identical to V2-a; tier=2 adds
  `TIER2_TRUNCATE_FILES` = {corporate_actions@announcement_date,
  dividends@declared_date}), work dir keyed by tier so tier-1/tier-2
  copies never collide, `still_excluded` + `truncated_files` in the
  payload. Driver `--tier` flag. 4 new fast tests (composition, tier-2
  cut vs tier-1-intact, future-effective-row-kept PIT semantic).
- The tier-2 run: `TIER2_PIT_CLEAN`, PASS on all 5 dates.

## What didn't

- No dead ends. The one thing worth flagging: the truncation A/B on this
  liquid-name / modern-date grid does NOT exercise the two residual
  non-PIT reads found in the trace — the `get_fundamentals` dateless IV
  fallback (fires only when PIT `get_iv_history` is empty) and
  `_split_adjust_option_premium`'s un-`as_of`'d corp-action read (fires
  only with option-premium parquets present). Both are dormant here, so
  the clean A/B does not certify them; they are recorded as standing PIT
  limitations, not swept under the PASS.

## How we fixed it

Measurement-only; no engine change. The caveat is resolved by evidence
(the tier-2 truncation PASS) plus a scoped audit of what truncation
cannot reach, recorded in plan §11.1 and findings §6.

## Evidence

- Pre-registration committed before any code: `fb755c6` (plan §11).
- `python -m pytest tests/test_freeze_replay.py -q` → 32 passed (4 new).
- `python scripts/run_freeze_replay.py amnesia --tier 2` (3m04s,
  `MarketDataConnector`, `use_credit_regime=False`, rail off):

  | as_of | verdict | rows | drops | corp_actions kept | dividends kept |
  |---|---|---|---|---|---|
  | 2022-06-15 | PASS | 15 | 9 | 46,240 | 46,178 |
  | 2023-11-15 | PASS | 20 | 4 | 48,497 | 48,452 |
  | 2024-08-06 | PASS | 16 | 8 | 49,678 | 49,640 |
  | 2025-04-15 | PASS | 3 | 21 | 50,713 | 50,683 |
  | 2026-05-01 | PASS | 22 | 2 | 52,513 | 52,494 |

  Full files: corp_actions 52,760 / dividends 52,737 → 6,000+ post-T rows
  dropped at the earliest date; identical A/B is load-bearing.
  Report: `data_processed/validation/freeze_replay/amnesia_report_tier2.json`
  (gitignored).

## Unresolved / handoff

- If a future pass wants tier-2 PIT *by construction* rather than
  by-grid: date-filter the `get_fundamentals` IV fallback (or drop it in
  favor of the PIT `get_iv_history` path) and thread `as_of` into
  `_split_adjust_option_premium`'s `get_corporate_actions` call. Both are
  engine changes — out of scope for the measurement-only phase; a D-series
  decision if pursued.
- With this closed, the validation phase's only remaining open items are
  the F-V6-1 premium bound and the E-13 capacity knee, both Theta-gated.
