---
id: data-tests-covered-call
title: Data-test PR-6 covered-call real-data coverage (W29-W32)
kind: verification
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-09
headline: Phase-2 round-2 PR-6 of the data-layer test audit. The covered-call ranker (the wheel's 2nd leg) was real-data-starved vs the put side — its only real-data assertion was the single DIS ex-div row. Adds W29 (CC banded/finite on real data, mirror of the put well-formed test), W30 (CC real earnings event-lockout, mirror W16), W31 (CC ex-div penalty SIGN lowers ev, controlled A/B grounded in the real DIS ex-div), W32 (CC EV-sign control HD +/UNH,AAPL -, mirror W15). Gaps found by the round-2 recon workflow. Test-only; trio/data untouched.
surface: [tests/test_data_to_engine.py]
---

## Goal
<!-- What we set out to do, and why. -->

Round-2 discovery (recon workflow) found the covered-call ranker
(rank_covered_calls_by_ev) is well-covered SYNTHETICALLY but real-data-starved
relative to the put side: the 2026-06-09 register added W15 (real EV-sign), W16
(real event-lockout), and the well-formed/finite-at-scale coverage to the PUT
ranker, but the CC leg received none of it — its only real-data assertion was the
single DIS ex-div plumbing row. PR-6 closes that.

## What we tried
<!-- Approaches, in the order we tried them. -->

`tests/test_data_to_engine.py` (+ a `_rank_cc` helper):
- **W29** `test_cc_clean_universe_output_is_well_formed` — loop UNIVERSE_24 through
  rank_covered_calls_by_ev; every produced row finite (ev_dollars/ev_per_day),
  banded (0<iv<3, prob∈[0,1]), strike>spot (OTM call — mirror of the put), Wilson-CI
  coherent, valid tier.
- **W30** `test_cc_real_earnings_event_lockout_fires` — JPM (real earnings 40d out) at
  49/63-DTE: gate ON → 0 rows, all drops gate=='event'; gate OFF → produces. Pins the
  real sp500_earnings.csv → EventGate → evaluate wire on the CC leg.
- **W31** `test_cc_exdiv_penalty_lowers_ev` — controlled EVEngine.evaluate A/B over the
  SAME forward distribution, toggling the REAL DIS ex-div (get_next_dividend → 0.75):
  with-exdiv ev < without (the call-only early-assignment penalty bites). The DIS row
  test only pinned the plumbing (expected_dividend==0.75), not the SIGN.
- **W32** `test_cc_ev_dollars_sign_controls` — HD +EV / UNH,AAPL -EV at FRONTIER (sign
  only, gate off). Mirror of W15.

## What worked

All 4 pass on real data; the data→engine suite stays green (19 passed, 12 xfailed).

## What didn't
<!-- The dead ends + WHY. -->

W31 can't be tested through the ranker (it always uses the real ex-div — no A/B
toggle), so the clean isolation is a direct EVEngine.evaluate call with the real DIS
dividend amount over a fixed forward distribution (still §2-clean: evaluate is the
authoritative EV path; the test asserts its behaviour, not a production bypass). The
recon's DTE-in/out-of-window comparison (-112 vs -49) is DTE-confounded and was NOT
used — the same-trade toggle (-94.69 vs -109.69) is the correct isolation.

## How we fixed it
<!-- The approach that shipped. -->

Test-only, real-data mirrors of the put-side coverage. W30/W32 are FRONTIER-tied;
W32's signs are robust to the pending ev_mean re-baseline (sign-only), W30 shares
W16's frontier-brittleness (JPM must stay within the earnings window) — captured in
the re-baseline memory note. The §2 routing/call-count for CC is already covered
synthetically (recon confirmed) so no real-data §2-spy was added.

## Evidence
<!-- Exact commands run, numbers. -->

Worktree off `origin/main 74685ef` (post PR-5 merge), provider `MarketDataConnector`.

- CC probe at FRONTIER: 10/10 names produced finite+banded; JPM 49/63-DTE gate on→0
  (event drops) / off→2; DIS ex-div toggle ev -94.69 → -109.69 (penalty -15.0); HD
  max ev +136 / UNH -144 / AAPL -157.
- `py -3.12 -m pytest tests/test_data_to_engine.py -m "not slow" -q` → **19 passed,
  12 xfailed, 1 deselected**. `ruff` clean.

## Unresolved / handoff
<!-- What's still open. -->

- **HOLD for review.** Next (round-2 register): PR-7 (realism-at-scale bands W33,
  vix→R11 content W35, liquidity W34), PR-8 (cross-file vol_iv↔ohlcv date consistency
  W36, data_integration rate divergence W37) + (E) issues for the engine-side parts.
- W30 (like W16) is JPM-earnings-window dependent — re-baseline session must confirm
  JPM stays in the 49/63-DTE earnings window or re-pick a near-earnings name.
