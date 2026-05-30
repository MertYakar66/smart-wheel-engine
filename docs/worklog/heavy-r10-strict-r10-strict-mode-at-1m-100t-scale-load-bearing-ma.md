---
id: heavy-r10-strict
title: R10 strict-mode at $1M/100t scale — load-bearing magnitude guard, never previously backtested
kind: verification
status: in-flight
terminal: D
pr:
decisions: []
date: 2026-05-30
headline: portfolio_delta_breach is the dominant binding D17 gate (96.7% of refusals in pilot); R10 is a "single-contract entry cap" for high-priced names (BKNG/AZO at $1M NAV), not an "accumulation cap"; loose tracker breached 10% per-name cap on 81.4% of pilot days
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

## Evidence
- Pilot artifacts: `docs/verification_artifacts/r10_pilot_2020-q1apr_summary.json`
  + `r10_pilot_2020-q1apr_raw_output.txt`. Full per-day rank logs +
  attempt logs live in `%TEMP%/r10_pilot/` (NOT committed per Sn
  throwaway-harness convention).
- Pilot wall-clock: 26.8min for 86 trading days (0.32 min/day = 18.7
  s/day) on the dev box. Projects 6.5h for full 5y run.
- Driver: `docs/verification_artifacts/r10_strict_driver.py` (~1300
  lines). Compiles + ruff-clean. Both `run_strict_vs_loose` and
  `analyze` modes exercised against pilot output.
- Full 5y run launched in background; results to be appended here +
  in the report doc (`docs/HEAVY_R10_STRICT_SCALE.md`) when it
  completes.

## Unresolved / handoff
- **Full 5y run results** to be appended to this fragment + the report
  doc when complete. Critical question for full run: does strict-
  mode's lower deployment DRAG performance during the 2021-2024 bull
  years (offsetting the Q1-2020 preservation seen in the pilot)?
- **Structural finding for Major Session triage**: the dominance of
  `portfolio_delta_breach` (96.7% of refusals in the pilot) may
  warrant a follow-up D-decision about whether the ±$300 / $100k NAV
  default is well-calibrated for the wheel strategy. The cap was set
  in D17 for a generic options book; a wheel book with assigned-
  shares legs has STRUCTURALLY long delta, and the cap may bind too
  early relative to the strategy's intended exposure profile. NOT
  in scope for HT-D — flagged as a finding.
- **R10 in entry-only mode** cannot prevent assignment-driven per-name
  accumulation breaches (loose tracker reached 25.36% per-name on
  2020-03-23 via assignments, not new opens). A future card could
  evaluate adding an assignment-time R10 check (refuse to wheel into
  CC on a name already > 10% NAV) or a closing-side R10 (force-close
  on a name that grew past the cap via assignment). Outside HT-D
  scope; surfaced as a finding.
