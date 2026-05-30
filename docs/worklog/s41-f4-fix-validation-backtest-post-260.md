---
id: S41
title: F4 fix validation backtest (post-#260)
kind: backtest
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** PR #260 (`realized_vol_widening_factor`, threshold
`rv30/rv252 >= 1.30`, max widening 1.15) replaces the rolled-back
Fix B1+C (PR #253). The fix's quoted S27 metrics (overall ρ +0.1819,
2022 ρ +0.364, COST 2022-04 unchanged) live in the PR body. S41
validates those claims end-to-end on three concentric layers — unit
reproduction of canonical F4 cases, full 2022 bear backtest, and
calm-regime signal preservation in 2023-2024 — and characterises
the calibration's behaviour across a 24-ticker × 36-monthly-date
grid.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
Engine: `origin/main` @ `56d8e5c` (post-PR #260 RV widening, post-PR
#262 R10 single-name cap). Three concurrent probes plus the in-repo
`backtests.regression.s27_ivpit_24t_100k` reproducer (identical
config to S27 — $100k, 24 SP500 tickers, 35-DTE / 25-Δ short puts,
wheel into CC on assignment, hold to expiry,
`require_ev_authority=False`, frictionless). Throwaway driver
scripts at `C:/Users/merty/AppData/Local/Temp/s41_f4_validation/`
(deleted before commit per the §3 convention).

**Path.** `WheelRunner.rank_candidates_by_ev` (the §2 route). Zero
edits to `ev_engine.py`, `wheel_runner.py`, `candidate_dossier.py`,
`forward_distribution.py` — validation only.

**Status.** Done. **Verdict: the fix is mechanically faithful to PR
#260's claims (named-case behaviour, calibration table, fire rate)
but does NOT close the dollar-damage gap on the named F4 cases —
and PR #260 §11 honestly admits this.** R10 single-name cap (PR
#262) is the actual damage-bounding mechanism for those cases. The
overall S27 ρ stays preserved (+0.1881 → +0.1819, −3.3% relative).
Calm-regime signal preserved across all 6 unit-control cells (factor
= 1.00 everywhere).

**Findings:**

- **(F1 — unit reproduction exact across all three named cases).**
  COST 2022-04-04 unchanged (rv30/rv252=0.9615, factor 1.0000,
  prob_profit 0.8333). UNH 2024-11-11 widens mildly
  (rv30/rv252=1.3607, factor 1.0121, ev_dollars +$114.53 →
  +$108.25 — exactly the PR #260 number). AAPL 2026-02-13 control
  byte-identical. All 10 dates of the canonical COST 2022-04
  unfolding-event window reproduce as
  `[prob_profit=0.8333, factor=1.0000] × 10`, mean EV +$127.35
  — exact match to S27 doc's pre-#260 cohort.
- **(F2 — calibration sample confirms PR #260's ~14% fire rate).**
  Sampled 449 cells (24 tickers × 36 monthly dates, 449 cleared
  event_gate). Fix fires on **54 cells (12.0%)** with max factor
  1.1239 (vs the 1.15 cap). 2022: 23.0% fire rate; 2023: 2.6%;
  2024: 11.0% — the fix concentrates its caution in the bear
  year exactly as designed.
- **(F3 — signal lags by ~30 days)**. The most informative
  diagnostic finding: the F4 fix is silent through the entire 10-day
  COST 2022-04-01 → 2022-04-14 unfolding-event window. By
  2022-04-14 the rv30/rv252 has crept up only to 1.0546 — still
  below the 1.30 firing threshold. By 2022-06-01 the same COST
  fires at factor 1.1239 (catching follow-on vol). The fix
  protects against **second-and-subsequent** vol-cluster events,
  not first-event idiosyncratic drawdowns. R10's per-name cap is
  the right tool for the latter.
- **(F4 — overall ρ preserved within margin)**. S27 backtest re-run:
  overall ρ +0.1881 → +0.1819 (−0.0062). 2022 ρ +0.3751 → +0.3638
  (−0.0113). 2023 ρ +0.1774 → +0.1795 (+0.0021). 2024 ρ +0.0782
  → +0.0693 (−0.0089). PR #260's hard gate (`ρ ≥ 0.15`) passes
  with margin. **No regime collapse on any year.**
- **(F5 — 2022 mean realized per ranked candidate drops 88%)**.
  $1.72 → $0.21. Counter-intuitive: more selectivity should improve
  mean realized, but mean_realized is computed across the full
  ranked universe (top-10/day) not the executed set. The fix
  reshuffles which candidates land in the top-10, and the new top-10
  in 2022 has slightly worse realized P&L per row. Strikes/premiums
  are byte-identical pre/post (BSM-derived) — the change is
  composition. Total dollar impact is small (~$2,900 over 1,936
  rows). The fix's value shows up in `executed_trades` (51 → 40,
  −22%) and `final_NAV` ($127,694 → $112,223, −12.1%, the
  documented trade-off in PR #260). On the **opener-eligible 2022
  subset** (1,100 rows where `ev_dollars > 0`), mean realized is
  **+$55.23 with 81.3% hit-rate** — the engine IS net-profitable
  on the trades it actually surfaces for opening; the losses
  concentrate in two clusters (COST 4/22 + MSFT 8/22).
- **(F6 — calm-regime preservation)**. 6 calm-control cells (AAPL,
  MSFT @ three calm 2023-2024 dates) all have factor=1.00 — no
  spurious caution. AAPL @ 2024-09-09 has `hmm_regime=bear` but
  factor=1.00 — confirms the post-PR-#253 diagnostic finding that
  HMM `bear/crisis` labels are vol-state labels, not tail-event
  predictors, and the RV-ratio signal is independent of (and more
  precise than) HMM.
- **(F7 — S27 snapshot byte-for-byte reproducible)**. Standalone
  S27 reproducer run (`backtests.regression.s27_ivpit_24t_100k`)
  produced 5,944-row rank_log + metrics.json matching the in-repo
  snapshot to 6+ decimal places across every aggregate / per-year /
  per-quartile field. The snapshot is current-engine-trustworthy.
- **(F8 — second worst-loss cluster: MSFT 2022-08)**. The 2022
  top-15 worst-realized is dominated by COST 2022-04 (11 entries,
  the canonical F4 case) PLUS MSFT 2022-08 (4 entries, prob_profit
  0.849, the August inflation surprise). MSFT 2022-08 is not in
  PR #260's named-case set but shows the same first-event pattern:
  pre-event `rv30/rv252 < 1.30`, F4 fix silent, R10 the only damage
  bound.

**AI handoff:**

- **For the F4 dossier:** PR #260's §11 entry in
  `docs/F4_TAIL_RISK_DIAGNOSTIC.md` is mechanically accurate. No
  doc edits needed.
- **For `docs/PRODUCTION_READINESS.md` §3 Blocker B1:** "partially
  closed" is the correct framing. S41 confirms the partial closure
  with quantified deltas (overall ρ −3.3%, NAV −12%, fire rate
  12%, calibration matches).
- **For the regression CI lane:** S32/S34/S35 snapshots remain
  pre-#260. Re-baseline as a follow-on (S42 candidate). The harness
  (`backtests.regression.*`) handles all four.
- **For future F4 work:** the COST 2022-04 first-event drawdown
  case is structurally not solvable by historical realized-vol
  signals (the vol cluster hadn't materialized yet). Closure would
  require a fundamental signal — earnings surprise probability,
  regulatory event detection — outside today's data layer.

Full doc: `docs/ENGINE_BACKTEST_S41_F4_FIX_VALIDATION.md`.

---
