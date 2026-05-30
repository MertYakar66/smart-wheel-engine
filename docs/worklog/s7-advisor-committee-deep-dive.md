---
id: S7
title: Advisor committee deep dive
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Verify S1's logged committee/ranker contract-mismatch
claim, and answer the trader question: do four advisors disagree
usefully, or is the committee expensive noise on retail short puts?

**Setup.** Bloomberg, offline charts, `as_of=2026-03-20`, 35-DTE /
25-delta, top-10 ROC names from S4 (CF, FIX, FDS, AJG, EXE, JBHT,
EG, MCK, HUM, BR), fed through
`advisors.integration.EngineIntegration.evaluate_trade` — naive
caller, then a corrected caller emulating the `/api/committee` path
(delta from spot/strike/IV, `ev_dollars → ev_pct`). Plus 12
synthetic probes varying one input at a time. No code changes.

**Status.** Done. All findings logged (no fix this session).
Code-level claims verified by Cowork-B against source; runtime vote
patterns/probe reactions as reported by the executor run.

**Findings:**

- **Committee structurally pinned at neutral.**
  `_determine_committee_judgment` leaves neutral only on
  `approve_count > total/2` or `reject_count > total/2` — i.e.
  ≥3 of 4 (`committee.py:331,337`). Three advisors default neutral
  on retail short puts, so the verdict never escapes neutral on
  realistic ranker output. **Logged.**
- **`filter_approved(min_approval_count=2)` blocks 100% of
  positive-EV picks.** Keyed on each trade's advisor
  `approval_count` (`integration.py:117-141`); max observed
  approves = 1 (Munger), so it returns 0 trades at thresholds 2, 3,
  and 4. **Logged.**
- **The `EngineIntegration` helper has type bugs `/api/committee`
  already fixed.** The helper passes `ev_dollars` straight into
  `expected_value` (documented "Expected return %"), rendering
  $247.76 of EV as "247.76%". The API path converts
  `ev_dollars → ev_pct` (`engine_api.py:871,883`) and rescales vix
  fraction→percent (`924-931`); `_build_advisor_input` does
  neither. Fix belongs in `_build_advisor_input` so all callers
  benefit, not duplicated per endpoint. **Logged.**
- **Per-advisor signal is real but discarded.** Negative EV → 2
  rejects, crisis regime → 2 rejects, earnings-in-expiry → 1
  reject; the >50% aggregator throws away sub-majority dissent. A
  `committee_judgment="elevated_concern"` on ≥2 dissents
  (escalation — §2-safe) would surface it. **Logged.**
- **Ranker emits no `delta`/`theta`/`gamma`/`vega`/`iv_rank`.**
  Forces the helper's −0.30 delta fallback. The ranker selects the
  strike via a chain `delta` column (`wheel_runner.py:899-907`)
  but emits no delta in its output. **Logged.**
- **"Areas of agreement" is substring keyword matching, not
  semantic synthesis.** **Logged.**
- **Simons is binary-on-EV; the others binary-off.** Simons
  strong-approves on high EV without distinguishing 5% / 10% /
  50%; net committee ≈ `Simons_thinks_EV_high ? lean : neutral`.
  **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds by structure — advisor committee is downgrade-only-advisory per CLAUDE.md §1 (interface layer, not on the EV path); not exercised live this pass because the public API changed.
  - Qualitative verdict: partial — `EngineIntegration` exists, but `evaluate_trade(ev_row)` and `filter_approved(rows, min_approval_count)` now both require additional positional args `portfolio_state` and `market_state` (signatures evolved post-S7). The naive-caller demonstration the original S7 exercised is no longer a single-arg call. The structural findings (committee structurally pinned at neutral; per-advisor signal real-but-discarded; Simons binary-on-EV) still apply at the source-code level — `_determine_committee_judgment` thresholds in `advisors/integration.py` are unchanged on `main`.
  - Numerical drift > 5%: not measured — the API signature shift blocks the original probe matrix.
  - Notes: ranked the 10 ROC names successfully (10/10 from the S4-derived watchlist); `EngineIntegration` instantiates clean. A future Sn could re-exercise with the new `portfolio_state` / `market_state` shape to re-confirm the "committee pinned at neutral" finding on real data.
