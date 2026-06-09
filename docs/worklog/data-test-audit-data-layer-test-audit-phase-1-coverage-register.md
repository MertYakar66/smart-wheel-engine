---
id: data-test-audit
title: Data-layer test audit Phase 1 — coverage register W14-W28
kind: research
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Deeper, test-coverage-focused data-layer audit on origin/main @ d0cdcde. Found Phase 2 (A)+(B) already shipped (#358/#366) so all of W1-W13 are closed/tracked; registered 15 NEW weaknesses W14-W28 (13 (T) landable, 1 (E)/1 (D) tracked) led by the untested #363 served-IV gate + the missing real-data EV sign control; corrected two precedent capability-map claims (credit is OFF the EV path; R9 sector uses a hardcoded DEFAULT_SECTOR_MAP). Phase 1 = doc + reproducible probe; HOLD before Phase 2.
surface: [docs/DATA_TEST_AUDIT_2026-06-09.md, scripts/audit_data_tests.py, FILE_MANIFEST.md]
---

## Goal
<!-- What we set out to do, and why. -->

We have pulled large datasets into the repo (sp500_ohlcv ~59M, sp500_vol_iv_full
~78M IV, fundamentals, credit, dividends, treasury) but coverage of the data
*itself* — (a) is the pulled data sane, (b) does each dataset convert into correct
engine output — is thinner than engine-logic coverage. This is a deeper, data-focused
round building on `docs/DATA_ENGINE_AUDIT_2026-06-07.md` (W1–W13) and the Phase-2
suites it spawned. Phase 1: recon + a weakness register evidenced on the bytes,
classified (T) test-only / (E) trio change / (D) producer change, ranked into PRs.
STOP for review before Phase 2.

## What we tried
<!-- Approaches, in the order we tried them. -->

1. Re-ran `scripts/audit_data_engine.py` on the worktree (origin/main `d0cdcde`) to
   refresh the W1–W13 byte-state post-#363.
2. Wrote a new probe `scripts/audit_data_tests.py` for the deeper figures the first
   script does not compute (served-vs-raw IV band, per-name OHLCV depth, GICS-11
   validity, S&P ladder, Altman-Z, the data→`rank_candidates_by_ev`→EVResult
   finite+sign check).
3. Fanned out a 4-agent read-only recon workflow (wiring confirmation, IV/OHLCV
   coverage, fund/credit/div/earn/e2e coverage, W1–W13 reconciliation) and
   independently re-verified every load-bearing claim against source.

## What worked

- The combination pinned a clean picture: **Phase 2 (A)+(B) already shipped**
  (`tests/test_data_integrity_bloomberg.py` + `tests/test_data_to_engine.py`, #358/#366),
  so W1–W13 carry no open *uncovered* test gap. The new value is the narrow set of
  genuinely-uncovered deeper gaps (W14–W28).
- The wiring pass corrected **two** precedent capability-map claims (verified at
  source): credit (`sp_rating`/Altman-Z) is OFF the EV path (no R0a gate; ranker
  `credit_mult` = FRED HY-OAS), and R9's sector cap uses a hardcoded
  `DEFAULT_SECTOR_MAP` (`risk_manager.py:1755`), not `gics_sector_name`.

## What didn't
<!-- The dead ends + WHY. This is the part that saves the next agent. -->

- A subagent reported "5,927 IV cells NULLed by #363" — **wrong**; that is the count
  of pre-existing NaNs. The gate NULLs **exactly 17** sub-3.0 cells on the committed
  monolith (`nulled_vs_raw_put=17`, served put min 3.127); the >10000 sentinel removes
  **0** (it lives only in the uncommitted deep panels). Always re-derive the load-
  bearing number — recorded the authoritative figure in the doc.

## How we fixed it
<!-- The approach that shipped. -->

Phase 1 ships two artifacts, held for review:
- `docs/DATA_TEST_AUDIT_2026-06-09.md` — capability map (+4 corrections), coverage
  map, W1–W13 reconciliation, register W14–W28, and a 5-PR ranked (T) plan.
- `scripts/audit_data_tests.py` — the reproducible probe behind the W14+ numbers.

No tests written (Phase 2). No trio edit, no data-file edit, no parallel-session file.

## Evidence
<!-- Exact commands run, numbers, links to raw artifacts. -->

Provider `MarketDataConnector`, frontier `2026-06-04`, HEAD `d0cdcde`.

- `py -3.12 scripts/audit_data_engine.py --universe audit` → W1–W13 reproduce;
  IV raw min 0.01 / max 769.273 / 17 rows in (0,3.0] / sentinel 0; OHLCV 4 NaN-price
  rows; rate_1m 185 negatives; dividends 82 epsilon-negatives; fingerprint 9/9.
- `py -3.12 scripts/audit_data_tests.py` → **served** IV put min **3.127**, ≤3.0 = **0**,
  >10000 = 0, nulled_vs_raw = **17** (gate verified on real served read); OHLCV depth
  min 7 / median 2117 / 17 names <504; NaN-price rows = BIIB 2020-11-06 & 2023-06-09,
  TPL 2019-05-16 & 2019-07-09; GICS = exactly the 11 canonical, 0 outside; eqy_dvd_yld
  [0.057%,10.77%], 95 NaN; sp_rating 21 distinct incl. CreditWatch suffixes (`A *-`…),
  52 NaN; Altman-Z [−5.43, 129.5], 75 NaN, 3 neg; e2e 0 non-finite, sign-bearing
  (XOM +112.86 / UNH −77.35, JPM event-locked-out).
- Source confirmations: #363 gate `data_connector.py:317-334` (band `(3.0,10000]`,
  monolith :227 + assembled :295); credit off-EV-path `wheel_runner.py:509-519` +
  `test_credit_rating_population.py` docstring; R9 `portfolio_risk_gates.py:372` →
  `risk_manager.py:1755/1760` `DEFAULT_SECTOR_MAP`; #366 PIT behaviour xfail
  `test_data_to_engine.py:307-345`.

## Unresolved / handoff
<!-- What's still open; what the next agent should look at next. -->

- **HOLD for review before Phase 2.** Phase 2 = land the (T) items as one PR per
  surface (PR-1 IV served-gate first), each additive + behaviour-pinning xfails
  (#366 lesson) + worklog + manifest.
- **Tracked, do NOT grab:** W27 (E — #363 does not clean the fundamentals-fallback IV
  path), W28 (D — `edge_vs_fair` ≡ 0 until a market-mid premium producer), plus the
  existing #354 (W2 PIT), #355 (W6 backfill), #357 (W10/W11 clamp) behind their xfails.
- Capability-map corrections C1/C2 mean the precedent doc + `audit_data_engine.py`
  `CAPABILITY_MAP` overstate credit (R0a) and GICS→R9 — a future tidy could align them.
