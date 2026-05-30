---
id: heavy-r10-strict
title: R10 strict-mode at $1M/100t scale — load-bearing magnitude guard, never previously backtested
kind: verification
status: completed
terminal: D
pr:
decisions: []
date: 2026-05-30
headline: full 5y run — strict loses to loose by −15.81pp; R10 fires 571× exclusively on BKNG/AZO (S44 prediction confirmed); portfolio_delta_breach dominates D17 refusals at 92.1% (R10 only 7.8%); strict opens 31 puts ALL in 2020 then freezes for 4 years; strict ahead 66.6% of days but crosses below loose 2023-11-01
surface: [docs/HEAVY_R10_STRICT_SCALE.md, docs/verification_artifacts/r10_strict_driver.py]
---

## Goal
Close the open question S44 §7 AI handoff flagged: every prior canonical
backtest (S27/S32/S34/S35/S38/S40/S43/S44) ran with
`require_ev_authority=False` on `WheelTracker`, so the **R10 single-
name notional cap** (`engine.portfolio_risk_gates.check_single_name_cap`)
— the doc-designated load-bearing magnitude guard for the engine's
`prob_profit` top-bin over-confidence (per
`docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` §10 +
`docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10 +
`docs/PRODUCTION_READINESS.md` §3 B1) — has **never** been exercised
in a backtest. Measure how often R10 (and the adjacent R7-R9 + portfolio
delta + Kelly gates) actually BIND under the canonical S38 setup
($1M / 100t / 2020-2024) when strict mode is ON, plus the NAV / return /
trade-count delta vs the same setup with strict mode OFF.

## What we tried
- **A/B harness inside ONE driver, sharing one daily SP rank call.**
  Two `WheelTracker` instances side-by-side: `tracker_loose`
  (`require_ev_authority=False`) and `tracker_strict`
  (`require_ev_authority=True`, `min_nav_for_trading=0`, attached
  connector for live mark-to-market NAV). Strict opens use
  `WheelTracker.consume_ranker_row` so D16 (token issue + consume with
  current_ev_dollars re-check) AND D17 (R9 sector cap + R10 single-
  name + portfolio delta + Kelly) all fire on every attempt. Loose
  opens use the direct `open_short_put` / `open_covered_call` path,
  matching S38/S44's baseline. **This eliminates harness-drift
  contamination** — comparing strict-vs-S38-published numbers would
  conflate "the throwaway S38 harness vs my new driver" with the R10
  effect. Same-driver A/B isolates R10 cleanly.
- **Driver imports** the canonical harness helpers (`friction_*`,
  `_spot_on_or_after`, `_next_business_day`, `_forward_replay_realized_pnl`,
  `ohlcv_sha256`) from `backtests/regression/_common.py` and the
  canonical `UNIVERSE_100` from `backtests/regression/universes.py` so
  per-day mechanics match S38/S44 byte-for-byte.
- **Per-attempt log** captured for every open attempt by introspecting
  `tracker._ev_authority_log` delta immediately before/after each
  attempt. Captures: outcome (`opened` / `refused` / `refuse_issue`),
  reason (`single_name_breach`, `portfolio_delta_breach`,
  `sector_cap_breach`, `kelly_size_exceeded`, `nav_exhausted`,
  `non_positive_ev`), and the gate's details bag (`post_open_name_pct`,
  `post_open_sector_pct`, `nav`). Plus `name_notional_pre` so a
  `single_name_breach` reject can be cross-referenced against existing
  per-name exposure.
- **Daily state series** captures end-of-day per-tracker NAV +
  per-name max %NAV + sector max %NAV + n_positions held. Powers the
  "how often WOULD R10 have bound in loose mode" counterfactual
  (loose tracker has the gate OFF but we can count days when its
  max-name notional exceeded the 10% cap).
- **Two-phase compute**: pilot first (2020-Q1+April, 86 days,
  26.8min) to validate correctness + measure preliminary firing rates,
  then full 5y (2020-01-02 → 2024-12-31, 1258 days, estimated ~6.5h).
- **Post-hoc analyzer** (`--analyze` mode on the driver) reads the
  on-disk artifacts and emits the markdown tables for the report
  doc. Decouples expensive backtest from cheap analysis pass.

## What worked
**Pilot completed cleanly** with major preliminary findings (full 5y
run in flight as of commit time; this section updates when it
completes):

- **Strict +7.7pp NAV vs loose** over Q1+April 2020 ($991,808 vs
  $914,543 final NAV; −0.82% vs −8.55% return). Strict mode preserved
  capital through the COVID drawdown by refusing 93% of put open
  attempts and 99% of CC open attempts.
- **Dominant binding gate is `portfolio_delta_breach`**, NOT R10.
  Pilot pilot data: 398 put + 124 CC `portfolio_delta_breach` refusals
  = 522 / 539 = **96.7% of all D17 refusals**. R10
  (`single_name_breach`) bound 17 times (3.1%); R9 (sector cap)
  and Kelly did not bind.
- **R10 fires exactly on the predicted high-priced names**: 15 BKNG
  refusals + 2 AZO refusals over 86 days. These are the only tickers
  in `UNIVERSE_100` where ONE 25-Δ short-put contract notional alone
  exceeds the 10% NAV cap at $1M (BKNG @ ~$1,990 = 19.9% of $1M; AZO
  @ ~$1,102 = 11.0%). R10 is a *single-contract entry cap* for these
  tickers, not an *accumulation cap*.
- **Counterfactual: loose tracker breached 10% per-name cap on 70 of
  86 pilot days (81.4%)**, peaking at 25.36% on 2020-03-23 (COVID
  bottom) with 45 names held. If R10 had been enabled on the loose
  tracker, it would have bound on essentially every day of the
  pilot — but for tickers OTHER than BKNG/AZO, those breaches happen
  via *post-assignment accumulation* (assigned-shares basis already
  > 10% of NAV by the time the tracker fully digests COVID), not
  *entry-time refusal*. R10 in entry-only mode (today's implementation)
  cannot prevent assignment-driven accumulation breaches.
- **§2 invariant CLEAN** in both modes: 0 opens with `ev_dollars <= 0`,
  0 opens with non-finite EV. `consume_ranker_row` refuses non-positive
  at issuance (R1-equivalent at the launch gate).
- **A/B design works**: same Spearman ρ (0.3911) on both sides
  confirms identical rank input. Only execution differs.

## What didn't
- **First pilot launch crashed on Unicode `→` (U+2192)** in a print
  statement on Windows cp1252 console (the date-range banner). Fix
  was trivial — replace with `->`. Same hazard exists in any future
  driver: keep ASCII-only in print statements OR `PYTHONIOENCODING=utf-8`
  (the user invocation pattern). Pinned in memory:
  `bash-env-var-expansion-trap.md` already warned about the related
  PowerShell-`$env:VAR`-in-Bash-tool trap; this is the print-side
  twin.
- **Initial design assumption that R10 would be the headline gate**
  was wrong. The pilot revealed `portfolio_delta_breach` dominates by
  ~30:1. R10 is still doing meaningful work (firing on BKNG/AZO
  exactly as S44 predicted) but the *quantitative* binding rate is
  much lower than the doc framing suggested. The qualitative framing
  (R10 = load-bearing magnitude guard) is correct only in the sense
  that R10 is the LAST line of defense for the single names where
  even one contract breaches; the portfolio_delta gate is the
  *first* line of defense (limits exposure across all names
  simultaneously).

## How we fixed it
N/A — observational verification work, not a code change. Driver
`docs/verification_artifacts/r10_strict_driver.py` shipped as the
deliverable; engine code is untouched per HT-D card rules
(READ-ONLY on `engine/`; bugs surface as findings for Major Session
triage, not in-line fixes).

## What worked (full 5y, 2020-01-02 → 2024-12-31, 7.00h wall-clock)

Full-run headlines (see `docs/HEAVY_R10_STRICT_SCALE.md` for the
complete report):

- **Final NAV: loose $1,405,794 (+40.58%) vs strict $1,247,668
  (+24.77%) = −15.81pp delta.** Strict loses the 5y race.
- **Strict was ahead on 66.6% of trading days** (peaked at +$149k /
  +18.5% NAV during COVID April 2020); crossed below loose
  permanently on 2023-11-01; ended the run −$158k behind.
- **R10 bound 571 times.** Exclusively on BKNG (331) + AZO (240) —
  S44 §7 AI handoff's prediction 100% confirmed.
- **`portfolio_delta_breach` dominates at 92.1% of D17 refusals**
  (6,704 of 7,276). R10 = 7.8%; R9 = 0.01% (one AZO fire); Kelly = 0.
- **Strict opens ALL 31 of its put trades in 2020. Strict opens
  ZERO puts in 2021, 2022, 2023, 2024.** The portfolio_delta cap
  saturates from assigned-stock long delta after the initial COVID
  acquisition phase and never frees room for new entries.
- **Loose tracker breached 10% per-name cap on 47.8% of trading
  days** (623 of 1,304); breach rate dropped monotonically from
  25.36% peak in 2020 → 4.64% peak in 2024 as the loose book
  naturally diversified.
- **§2 invariant CLEAN in both modes** (0 opens with non-positive
  or non-finite EV).

## Evidence
- Driver: `docs/verification_artifacts/r10_strict_driver.py`
  (~1,400 lines). Compiles + ruff-clean. Both `run_strict_vs_loose`
  and `analyze` modes exercised end-to-end.
- Pilot artifacts (Q1+April 2020, 86 days, 26.8min):
  `docs/verification_artifacts/r10_pilot_2020-q1apr_summary.json` +
  `r10_pilot_2020-q1apr_raw_output.txt`.
- Full-run artifacts (5y, 1,258 days, 7.00h):
  `docs/verification_artifacts/r10_full_2020-2024_summary.json` +
  `r10_full_2020-2024_raw_output.txt` (trimmed; HMM `RuntimeWarning`
  bookkeeping stripped) + `r10_full_2020-2024_analysis.txt` (the
  `--analyze` output).
- Full per-day rank logs + attempt logs + tracker states +
  equity curves live in `%TEMP%/r10_full/` (NOT committed — they are
  ~10 MB of CSV/JSON per the canonical Sn throwaway-harness
  convention). Re-runnable from the driver: `python
  docs/verification_artifacts/r10_strict_driver.py --start
  2020-01-02 --end 2024-12-31 --out-dir <some_temp_dir>/r10_full`.

## Unresolved / handoff
- **`portfolio_delta_breach` is STRUCTURALLY MISCALIBRATED for the
  wheel strategy.** A wheel book has long-delta from assigned stock
  by design; the ±$300 / $100k NAV cap (set in D17 for a generic
  options book per #154 C4) freezes the strict tracker after the
  assigned-stock book exceeds ~$3,000 delta-dollars at $1M NAV
  (which happens after ~21 assigned positions at any realistic
  spot). **Recommended follow-up D-decision**: either exclude
  stock-leg delta from `check_portfolio_delta`'s aggregation, scale
  the cap with deployed (not initial) capital, or set a wheel-
  specific cap multiplier. Out of HT-D scope; flagged for Major
  Session triage.
- **R10 in entry-only mode** cannot prevent assignment-driven per-name
  accumulation breaches (loose tracker reached 25.36% per-name on
  2020-03-23 via assignments, not new opens). A future card could
  evaluate adding an assignment-time R10 check (refuse to wheel into
  CC on a name already > 10% NAV) or a closing-side R10 (force-close
  on a name that grew past the cap via assignment). Outside HT-D
  scope; surfaced as a finding.
- **The deployment-story answer is regime-dependent.** Strict won
  every bear / drawdown phase (peaked +$149k ahead in COVID-April-
  2020); loose won the 2023-2024 bull recovery. Over the full 5y
  window, loose wins by −15.81pp. A regime-aware strict/loose
  switching strategy is a separate (large) research surface — out
  of HT-D scope.
- **R10's binding rate is non-stationary across the run.** Loose's
  max-name %NAV monotonically decreased from 25.4% (2020) to 4.6%
  (2024) as the book naturally diversified. **R10 is most useful in
  early-phase / drawdown-shock phases when concentration is acute;
  less useful in steady-state when the book has self-diversified.**
  Production implication: R10 adds the most defensive value in the
  first 1-2 years after a fresh deployment, less in years 3+.
