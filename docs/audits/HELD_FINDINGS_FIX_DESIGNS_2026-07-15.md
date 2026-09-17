# Held-Findings Fix Designs — 2026-07-15 (CMD 10, EXEC-2)

Each top HELD finding from `docs/audits/WEAKNESS_AUDIT_2026-07-15.md` (branch
`claude/adoring-heisenberg-4gvoT`) is turned into an **operator-ready, PROVEN
fix target**: a reproduction test asserting the *correct* behavior — currently
failing on real engine code, so marked `@pytest.mark.xfail(strict=True)` (green
now, flips to an XPASS→failure alert the moment the operator lands the fix) —
plus a root-cause · minimal invariant-safe fix · blast-radius note.

**These are READ-MOSTLY repros. No `engine/` file was edited — the operator
applies the actual fix.** All findings touch the CI-gated trio / invariant /
risk-math, so they are held for operator review, not auto-dispatched.

Verification (this branch, `pytest tests/test_held_finding_*.py -q`):
`5 xfailed` normally; `5 failed` under `--runxfail` (proving each repro genuinely
fails on current engine code). Drop the `xfail` marker when each fix lands.

---

## F1 — roll_put/roll_call bypass the EV-authority token + D17 concentration caps  `[INV, highest priority]`
- **Where:** `engine/wheel_tracker.py` `roll_put` (~L1066/1031), `roll_call` (~L1164).
- **Repro:** `tests/test_held_finding_roll_ev_bypass.py::test_roll_put_must_enforce_single_name_cap_on_rolled_leg` (xfail).
- **Root cause:** `roll_put`/`roll_call` open a genuinely new short leg (new
  strike/premium/expiry/iv) but are wired through **neither** the D17
  concentration gate **nor** the D16 EV-authority token that `open_short_put`
  (token L667/L702, D17 L696-727/751) and `open_covered_call` (L1385/1420, L1454)
  enforce. On a `make_live_book_tracker` (production default,
  `enforce_single_name_cap=True`, 10% NAV cap): open a put at ~9% NAV, then
  `roll_put(new_strike=~2×)` → single-name notional ~19% NAV, applied with **zero
  cap check** and no fresh `evaluate`/token (contradicting CLAUDE.md §2).
- **Minimal invariant-safe fix:** before mutating `pos`, call
  `_evaluate_d17_hard_blocks(proposed_notional=new_strike*100)` (symmetric for
  `roll_call`), and in strict mode consume a fresh EV-authority token for the new
  strike/expiry; on reject, return `None` leaving the old leg intact (atomic — no
  partial close). Refusal-only, never touches `ev_raw`/`ev_dollars`.
- **Blast radius:** only the two roll methods; `suggest_rolls` and the open paths
  are unaffected. (The same hole exists in `roll_call` and the strict-mode token
  path; this repro isolates the production-default single-name bypass in `roll_put`.)

## F2 — dollar-gamma P&L is 100× understated in VaR and stress  `[INV]`  ✅ RESOLVED
- **Status (2026-07-21):** FIXED on `main` via **PR #496** (audit #1) — the `/100`
  was dropped, so `gamma_dollars = gamma*multiplier*spot*spot` (`engine/risk_manager.py:365`).
  The held repro `test_held_finding_dollar_gamma_100x.py` was removed (it began
  XPASS-ing under `xfail(strict)`); the corrected behavior is now pinned by
  `tests/test_risk_manager.py::TestGammaDollarsConvention::test_gamma_dollars_carries_contract_multiplier`.
  The design below is retained as the historical record of the finding.
- **Where:** `engine/risk_manager.py` `calculate_portfolio_greeks` L363 (+ every
  gamma-P&L consumer: hist-VaR L651, parametric L542, stress L1259/1303/1349/1376,
  limit L1479).
- **Root cause:** `gamma_dollars = gamma*multiplier*spot²/100` (L363, **WITH /100**)
  while `delta_dollars` (L362) has **no /100**. Every consumer then pairs
  `gamma_dollars` with a *fractional* move squared (e.g. `0.5*gamma_dollars*ret²`),
  so the convexity term is exactly **100× too small** while delta is
  self-consistent. Verified: a 10-lot short $95 put (S=100, T=0.10, σ=0.30) reports
  a −10% gamma-P&L of −$17.69 vs the analytic −$1768.69.
- **Minimal invariant-safe fix:** drop the `/100` at L363 → `gamma_dollars =
  gamma*multiplier*spot²`; then `0.5*gamma_dollars*ret² = 0.5*gamma*shares*dS²`
  (correct). No consumer changes needed (all already use the fractional convention).
- **Blast radius:** raises reported gamma-P&L/VaR/stress 100× for all option books
  → re-check the `max_portfolio_gamma_dollars` limit (default 50000), the L49
  docstring, and the dashboard scale. Add the contract's delta+gamma+vega-vs-reprice
  regression per `GREEKS_UNIT_CONTRACT.md`.

## F3 — HMM `bull_quiet` label by within-window rank, not return sign → 1.25× up-size on a down regime  `[INV]`
- **Where:** `engine/regime_hmm.py` `_label_states(K=4)` L363-364 (+ `fit` sort
  L225-232, `position_multiplier` L302-312).
- **Repro:** `tests/test_held_finding_hmm_bull_quiet.py` — `test_negative_mean_top_state_is_not_bull_quiet`, `test_down_regime_position_multiplier_not_upsized` (both xfail).
- **Root cause:** `_label_states(K=4)` unconditionally returns
  `["crisis","bear","normal","bull_quiet"]` by positional lookup after `fit` sorts
  states ascending by `means[:,0]-0.5*stds[:,0]`; so the least-negative state is
  **always** `bull_quiet` and `position_multiplier` weights it 1.25 with **no sign
  gate**. A steady low-vol decline (all state means negative) therefore up-sizes
  short-put EV 1.25× on the calm days a seller feels safe — the assignment trap.
- **Minimal invariant-safe fix:** gate the `bull_quiet`/`bull` label (and/or the
  >1.0 weight) on the state's mean sign — a state with `means[k,0] < 0` falls back
  to `normal`/≤1.0. (Simplest robust variant: cap `position_multiplier` at 1.0 when
  the already-computed `hmm_realized_return_252d_ann` is negative.)
- **Blast radius:** a sign gate only ever **demotes** negative-mean states, never
  promotes → cannot loosen sizing (downgrade-only, §2-safe). Labels feed the
  multiplier weighting + any diagnostic/regime-cache rows.

## F4 — Backtest lookahead: IV fallback substitutes today's snapshot IV when PIT IV is missing  `[INV]`
- **Where:** `engine/wheel_runner.py` puts ranker L1671-1701 (siblings:
  covered-call 3083-3085, strangle 3716-3718).
- **Repro:** `tests/test_held_finding_iv_fallback_lookahead.py::test_ranker_does_not_use_snapshot_iv_at_historical_as_of` (xfail).
- **Root cause:** when `_resolve_pit_atm_iv` returns `None` (no PIT IV axis), the
  fall-through does `iv_raw = fundamentals.get("implied_vol_atm")` with **no `as_of
  is not None` guard**. Per `data_connector.py:1565` that value is the CURRENT
  snapshot column (`30day_impvol_100.0%mny_df`, no date axis), so a dated backtest
  prices BSM strikes/synthetic premium off 2026 IV — a genuine lookahead leak.
- **Minimal invariant-safe fix:** gate the snapshot fall-through on `as_of is
  None`; when `as_of` is set and PIT IV is `None`, **drop the name** (append a
  `data` drop reason) rather than substituting snapshot IV. Strictly conservative,
  refuse-only.
- **Blast radius:** the same one-line guard is needed at the two sibling rankers;
  historical runs may drop names that were previously look-ahead-priced (correct).
  Live ranking (`as_of=None`) is unchanged.

---

*Repro tests authored + independently verified by EXEC-2 (CMD 10). No engine code
was modified; the fixes above are for operator review (all `INV`).*
