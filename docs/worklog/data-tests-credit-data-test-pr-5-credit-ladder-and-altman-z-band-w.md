---
id: data-tests-credit
title: Data-test PR-5 credit ladder and Altman-Z band (W24)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Phase-2 PR-5 (final register PR) of the data-layer test audit. Adds W24 — sp_rating ladder validity after stripping the CreditWatch suffix (' *-'/' *+'), and an Altman-Z plausibility band. Credit is OFF the EV-authoritative path (capability C1: feeds the legacy heuristic + display only, never EVEngine.evaluate), so these are display-severity. Completes the W14-W27 (T) register items; W28 (D) stays tracked. Test-only; trio/data untouched.
surface: [tests/test_data_integrity_bloomberg.py]
---

## Goal
<!-- What we set out to do, and why. -->

Fifth and final Phase-2 PR for the register's (T) items. PR-5 = credit content
validity (W24). LOW priority because credit is OFF the EV path (C1) — but cheap and
catches a producer regression in the rating/Altman columns.

## What we tried
<!-- Approaches, in the order we tried them. -->

`tests/test_data_integrity_bloomberg.py`:
- **W24** `test_credit_sp_rating_is_valid_ladder` — every sp_rating, after stripping
  the CreditWatch suffix (`' *-'`/`' *+'`), is in the S&P ladder (AAA..D) or a
  non-rated sentinel (NR). The raw field carries suffixed values ('A *-', 'CCC+ *+')
  a naive parse rejects; verified all 21 distinct values normalise cleanly.
- **W24** `test_credit_altman_z_plausible_band` — altman_z_score in [-10, 200] with a
  bounded negative count (≤10). The 2 values >100 are known off-EV artifacts
  (financials/REITs where Altman-Z is not meaningful); the wide band accepts them
  while catching a producer regression flooding negatives or absurd magnitudes.

## What worked

Both pass; integrity suite green (42 passed, 1 xfailed).

## What didn't
<!-- The dead ends + WHY. -->

n/a — the suffix-strip regex + band were verified on the real file before writing
(all 21 ratings → ladder∪{NR}; Altman [-5.43, 129.5], 3 neg, 2 >100).

## How we fixed it
<!-- The approach that shipped. -->

Test-only. Wide band (accepts current data, catches gross regressions) per the (T)
LOW classification — the credit-off-EV-path fact (C1) means tighter pins aren't
warranted. No engine/data change.

## Evidence
<!-- Exact commands run, numbers. -->

Worktree off `origin/main 7bc8b85` (post PR-4 merge), provider `MarketDataConnector`.

- 21 distinct sp_rating values incl. CreditWatch suffixes → 0 outside the ladder
  after strip; altman_z [-5.43, 129.52], 3 negative, 2 >100.
- `py -3.12 -m pytest tests/test_data_integrity_bloomberg.py -q` → **42 passed,
  1 xfailed**. `ruff` clean.

## Unresolved / handoff
<!-- What's still open. -->

- After this PR the register **W14–W27 (T) items are all landed** (PR-1..PR-5).
  Remaining register items are tracked-not-grabbed: W28 (D, edge_vs_fair / market-mid
  premium), plus the (E)/(D) issues #369 (fundamentals-fallback IV), #372 (R9→GICS),
  and the pre-existing #354/#355/#357.
- Next (autonomous): a fresh discovery pass on surfaces the 2026-06-09 round didn't
  reach (liquidity, vix, covered-call path, deep-read) + a realism-at-scale band test
  (the slow full-universe test asserts finiteness but not the output bands).
