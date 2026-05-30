---
id: HT-B
title: HT-B — PIT realism vs actuals (in-sample 2022-2024 + OOS 2020/2025)
kind: verification
status: completed
terminal: B
pr:
decisions: []
date: 2026-05-30
headline: Robust finding — mid-high bins (0.85,0.95] uniformly miscalibrated −11 to −17pp across 3 windows (n=68-552 per bin). Top-bin (0.95,1.0] sample-poor (n=8-67); apparent regime-dependence is hypothesis-generating only. Engine-exact pnl definition flips in-sample top bin to OK; ~12pp of headline OTM-convention "over-confidence" is methodology artifact. H3 verdict CORRECTED post-review from SUPPORTED → INCONCLUSIVE (sampling variance + HT-C cross-validation).
surface: [docs/HEAVY_PIT_REALISM.md, docs/verification_artifacts/pit_realism_driver.py]
---

## Goal

Test whether the post-#249 engine's `prob_profit` forecasts match what
actually happens. Specifically: does the **top-bin (0.95, 1.0]
over-confidence** documented in `PROB_PROFIT_CALIBRATION_2026-05-28.md`
(−15 to −17pp delta across 10 prior configs) still hold on
`main @ 56c671d`, AND does the in-sample (2022-2024) vs out-of-sample
(2020 crisis + freshest 2025) split show meaningfully different
miscalibration?

The prior calibration analysis pooled across 10 backtests but didn't
explicitly partition in-sample vs OOS. This re-tests on a fresh,
independent dataset across multiple regimes — confirming, refuting, or
updating the structural claim.

## What we tried

1. **Engine surface mapping.** Read `engine/wheel_runner.py`
   `rank_candidates_by_ev` signature: `as_of` is passed PIT-correctly
   to the connector + IV history lookup (`_resolve_pit_atm_iv`); the
   output row carries `prob_profit` as the engine's forecast. The
   put-side ranker row schema includes `ticker, spot, strike, premium,
   iv, ev_dollars, ev_per_day, prob_profit, distribution_source` —
   everything the calibration analysis needs in one call.

2. **Universe + dates.** Used the canonical `UNIVERSE_100`
   (`backtests/regression/universes.py`) for direct comparability with
   the S34/S38/S40 columns of the calibration doc. Sub-window dates:
   in-sample = first business day of each month 2022-01 → 2024-12 (36
   dates); OOS 2020 = bi-weekly Mondays Jan-Dec 2020 (26 dates,
   covers COVID Feb-Apr spike); OOS fresh = bi-weekly Mondays
   2024-09 → 2026-02-02 (38 dates, bounded by `as_of + 35d ≤
   data_end 2026-03-20`).

3. **Driver build.** `docs/verification_artifacts/pit_realism_driver.py`
   — single-purpose: for each (window, as_of), call ranker, write
   long-form rows (engine forecast + realised OTM at +35d) to a
   resumable CSV, then aggregate to per-window calibration tables on
   the established 7-bin scheme and pre-declared standard
   (|Δ|≤5pp ✅ / 5-10pp ⚠ / >10pp ❌). Bloomberg CSV column-rename
   quirk handled locally (CSV `high` is the true close) so the
   driver doesn't have to call the connector per (ticker, as_of) pair.

4. **Smoke test on one as-of.** as_of=2024-06-03 returned 88 candidates
   in 14.9 s, prob_profit range 0.63-0.91, realised closes resolved
   correctly. CSV column-rename cross-checked: AAPL 2026-03-20 close
   = 247.64 exactly matches `connector.get_ohlcv`.

5. **Full sweep.** 100 dates × 100 tickers in 1,777 s (~30 min) →
   4,581 rows captured, 4,537 with realised close (44 at the
   2026-03-20 data cliff).

6. **Methodological supplement.** Realised on TWO definitions
   side-by-side: (a) `realized_otm = close ≥ strike` (matches the
   prior calibration-doc convention); (b) `realized_pnl_positive =
   close ≥ strike − premium` (the engine's EXACT prob_profit
   definition). The two diverge by the shallow-ITM-but-still-
   profitable band; the comparison shows how much of the OTM-
   convention "over-confidence" is methodology artifact vs real
   engine miscalibration.

## What worked

- Routing the engine call through `rank_candidates_by_ev(as_of=...)`
  rather than building a custom EV pipeline — re-uses every existing
  PIT guard in the engine (OHLCV cutoff, IV-history `end_date=as_of`,
  fundamentals PIT, earnings calendar PIT). Zero new PIT logic in
  the driver.
- Pre-loading the OHLCV close series into memory once
  (`load_ohlcv_close()`, 1.1 s for 503 tickers) so the realised-
  close lookup is just an indexed pandas slice — keeps the per-as-of
  cost dominated by the ranker itself, not by re-reading the CSV.
- Resumable CSV (append-mode write per as-of + `already_done()` skip
  set) — a crash mid-run loses no completed dates.
- Pre-registering 3 hypotheses (H1 structural / H2 in-sample tuning /
  H3 post-#249 shift) BEFORE the driver ran kept the verdict
  intellectually honest. The H1-PARTIAL-REFUTED / H2-REFUTED /
  H3-SUPPORTED outcome was not the one I'd have guessed up front.

## What didn't

1. **First run hit a Windows cp1252 encoding error on the first
   non-ASCII char (`→`) in the progress lines.** Fix was the
   well-known `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`
   guard that several other scripts in this repo already use
   (`scripts/pull_edgar_earnings.py`, etc.). **Pattern: any driver
   that prints non-ASCII needs this stanza before its first print.**

2. **Cosmetic loop-var bug.** Initial date-count header showed `0`
   per window — `for _, _ in dates if _ == w` rebinds both names to
   the loop var so the `if` compares the as_of string to the window
   name (always False). Fixed before commit; the actual per-as-of
   loop was unaffected.

3. **Bonus analyses added MID-RUN don't auto-fire on the running
   process.** I extended `aggregate()` with 4 bonus tables (within-
   2020 phase, by distribution_source, in-sample-vs-OOS contrast,
   per-year-in-sample) AFTER the background driver started.
   These weren't in the running process's memory, so they were
   skipped at end-of-run. Resolution: re-invoked `aggregate()` from
   a fresh `python -c "from pit_realism_driver import aggregate;
   aggregate()"` to get all the bonus tables. The CSV is the source
   of truth; the aggregate function is stateless.

## How we fixed it

The substantive method is unchanged from the smoke test. The
mechanical fixes (encoding stanza, loop-var rename, ASCII-only
printable for the print headers) shipped before the full run.

For the analysis side, the regime-dependent finding required:
- **A diagnostic step on the dramatic OOS-fresh −35pp result** —
  pulled the 8 candidates by hand from the CSV. Found AVGO ×3 +
  ADI ×1 sector-concentration in the 4 losses. Documented that
  R10 single-name cap would block this in production. The
  driver intentionally does NOT apply R10 (it's a forecast
  calibration test, not a deployed-system test).
- **The methodological supplement** (OTM vs exact-pnl). Showed
  that ~12pp of the in-sample top-bin "over-confidence" is
  methodology artifact (shallow-ITM-but-still-profitable).
  In-sample top-bin Δ flips OK under the exact-pnl definition.
  Future calibration analyses should report both.

## Evidence

- **Driver runtime:** 1,777 s (~30 min) on Bloomberg connector.
- **Data captured:** 4,581 rows; 4,537 with realised close.
- **Per-window top-bin Δ (OTM convention):**
  in-sample = −9.07pp (n=8, WARN);
  OOS-2020 = −6.81pp (n=67, WARN);
  OOS-fresh = −35.00pp (n=8, MISCAL, dominated by AVGO ×3 + ADI ×1
  sector-concentration that R10 would mitigate in production).
- **Per-window top-bin Δ (engine EXACT pnl > 0 definition):**
  in-sample = **+3.43pp (flips OK)**; OOS-2020 = −5.31pp (WARN);
  OOS-fresh = −35.00pp (unchanged — losses were deep ITM).
- **Mid-high bins (0.85, 0.95]:** uniformly miscalibrated
  −11 to −17pp across all 3 windows. This is the most robust
  structural finding.
- **§2 invariant:** 0 non-finite EV rows; R1a guard intact.
- **Mechanism finding (bonus):** `empirical_overlapping` is much
  better-calibrated in the top bin than `empirical_non_overlapping`
  (−7.16pp vs −19.05pp pooled). Hypothesis-generating for a
  research follow-up.

Full per-window tables (8 bins × 3 windows × 2 metrics) +
hypothesis verdicts + diagnostic candidate listing in
`docs/HEAVY_PIT_REALISM.md`.

## Unresolved / handoff

- **Verifier-driven correction (2026-05-30):** the original
  write-up labelled H3 SUPPORTED based on the in-sample −9.07pp
  vs calibration-doc S34 −15.05pp delta. A paired-Session
  read flagged two issues: (1) at n=8 the binomial 95% CI on
  the in-sample top bin overlaps S34's reading — the delta is
  inside sampling variance; (2) HT-C (paired terminal,
  in-flight) reports "10/10 calibration re-derivations match
  published values" on the post-#249 engine — the apples-to-
  apples reproducibility test directly contradicts SUPPORTED.
  H3 is now INCONCLUSIVE. The lesson: at small top-bin n, a
  single backtest is not enough to claim engine drift; cross-
  validation with an apples-to-apples re-derivation is the
  right primary evidence.
- **OOS-fresh top-bin n=8 with sector concentration** wants a
  follow-up larger-n test on a wider universe to firm up the
  −35pp magnitude. The current data point is real (the 3 ITM
  losses are not noise — ADI alone lost $4,122 per contract)
  but the magnitude is sensitive to small-n AND to R10-being-
  off in the candidate-level analysis. R10 would have blocked
  the 2nd and 3rd AVGO entries in production.
- **The empirical_overlapping vs non_overlapping calibration gap**
  is the most actionable new finding from this study. A non-§2
  follow-up could test whether preferring overlapping for top-bin
  candidates (or applying a post-hoc shrinkage based on source)
  improves calibration without touching the EV authority. Caveat:
  the gap is partly window-confounded (overlapping is used more
  in OOS-2020 which is also the better-calibrated window) — a
  controlled test would need to use the same window with both
  sources where possible.
- **The 2020a pre-COVID phase** is the canonical empirical case
  of "regime change inside the forward window" — mid-bin Δ
  −47 to −65pp. The F4 fix mechanism (RV30/RV252 widening) is
  designed for elevated-realized-vol regimes, but pre-COVID
  was CALM realised vol with a crisis in the +35d forward
  window — so F4 would NOT have fired. R10 magnitude bounding
  is the right guard for this failure mode (consistent with
  the calibration doc and PROD_READINESS framing).
- Read-only on `engine/` per HT-B card rules: bugs surfaced here
  go to the board as findings for Major Session triage, NOT
  fixed inline.
