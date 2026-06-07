---
id: pricer-failloud-guards
title: Engine pricer fail-loud guards (R8 vectorized S<=0/K<=0; R37 BAW call r<=0 NaN)
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-02
headline: Two input-validation hardenings to engine/option_pricer.py — the three vectorized BS pricers now raise on any S<=0/K<=0 element (mirroring the scalar contract) instead of silently emitting NaN, and the BAW American-call branch short-circuits to European for r<=0 (was a div-by-zero NaN at r=0). Behaviour on valid inputs is byte-for-byte unchanged; §2-adjacent guard only.
surface:
  - engine/option_pricer.py
  - tests/test_option_pricer.py
  - tests/test_advanced_quant.py
---

## Goal
<!-- What we set out to do, and why. -->

Add INPUT-VALIDATION guards to the option pricer so invalid inputs fail
loudly instead of silently producing NaN that flows into the decision
layer. This is §2-adjacent (the pricer feeds `EVEngine.evaluate` via the
ranker's batch Greek/exposure path), so the bar is: change NOTHING about
behaviour on VALID inputs — only add guards that raise on truly-invalid
inputs (S<=0 or K<=0), matching the existing scalar `_validate_inputs`
contract.

Two concrete defects:

- **R8 (fail-loud):** the scalar pricers (`black_scholes_price`, etc.)
  call `_validate_inputs`, which raises on `S<=0`/`K<=0`. The three
  vectorized pricers (`vectorized_bs_price`, `vectorized_bs_delta`,
  `vectorized_bs_all_greeks`) did NOT validate, so a single bad strike
  (`K<=0`) or spot (`S<=0`) from a data gap silently became
  `log(<=0) -> NaN` inside the batch calculation.
- **R37 (correctness):** `american_option_price`'s call branch, with
  `r<=0` and `q>0`, skipped the `q<=0` short-circuit and fell into
  `_baw_critical_price_call`, where `k = 1 - exp(-r*T) = 0` at `r=0`
  caused `4*M/k` to divide by zero -> NaN (and produced a spurious
  early-exercise premium for `r<0`).

## What we tried
<!-- Approaches, in the order we tried them. -->

1. Read `engine/option_pricer.py` end-to-end; reproduced both defects
   directly before touching code:
   - `american_option_price(100,100,0.5, r=0.0, 0.20, "call", q=0.05)` ->
     `nan` (RuntimeWarning: invalid value in scalar divide at the
     `discriminant = (N-1)**2 + 4*M/k` line); `r=-0.01` -> `8.80` vs a
     European value of `4.22` (spurious premium).
   - The three vectorized fns silently returned NaN for `K=-5` / `S=0`
     array elements.
2. For R8, considered per-element masking (NaN-out only the bad rows)
   vs. fail-loud raise. Chose fail-loud raise on ANY non-positive element
   to match the scalar contract and the project's "fail loud, don't
   silently NaN" stance — silently masking would hide the upstream data
   gap.

## What worked

- **R8 — one shared validator `_validate_vectorized_inputs(S, K)`**
  added next to `_vectorized_intrinsic`, called at the top of all three
  vectorized fns right after `S`/`K` are coerced to arrays (and before
  any `np.log`/`is_deterministic` work). Raises `ValueError` if
  `np.any(S <= 0)` or `np.any(K <= 0)`, with a message mirroring the
  scalar `_validate_inputs` phrasing ("Spot price S must be positive,
  got ..." / "Strike price K must be positive, got ..."). The guard is
  placed BEFORE the `is_deterministic = (T_raw<=0)|(sigma_raw<=0)` mask
  and does not touch that path, so the deterministic-expiry handling for
  valid S,K is untouched.
- **R37 — a single `if option_type == "call" and r <= 0: return european`
  short-circuit** added immediately after the existing
  `call and q <= 0` early-out (and above the existing
  `put and r <= 0` early-out). A call with `r<=0` is never optimally
  exercised early — the only incentive to exercise a call early is to
  capture a dividend, and that requires `r > 0` to make forgoing the
  interest on the strike worthwhile — so American == European here. The
  put branch is untouched.
- **Value-equality proof of no behaviour change on valid inputs.** The
  R8 tests assert each vectorized fn matches the scalar pricer to `<1e-9`
  on a 3-element all-valid array (mixed calls/puts), and the R37 control
  test confirms the `r>0, q>0` BAW path still produces
  `American >= European`.

## What didn't
<!-- The dead ends + WHY. This is the part that saves the next agent. -->

- **Per-element NaN masking for R8** (keep the bad rows as NaN, compute
  the rest). Rejected: it hides the upstream data gap that the scalar
  contract is explicitly designed to surface, and the project stance is
  fail-loud, not silent-NaN.
- **Touching the `is_deterministic` mask / safe-value substitution
  (`T<-1.0`, `sigma<-0.2`) for the S<=0/K<=0 case.** Rejected: that
  machinery exists for the T<=0/sigma<=0 expiry path and must stay
  byte-for-byte unchanged for valid S,K. The guard sits strictly before
  it and only checks S/K positivity.
- **Returning the European price by mutating the existing `put and r<=0`
  block** to also handle calls. Rejected: a separate explicit call-branch
  line is clearer and keeps the put branch literally untouched.

## How we fixed it
<!-- The approach that shipped. -->

Single commit on `claude/pricer-failloud-guards` (branched from
`origin/main` @ `e1d7453`):

- `engine/option_pricer.py`
  - Added `_validate_vectorized_inputs(S, K)` (raises `ValueError` on any
    `S<=0`/`K<=0` element, scalar-style message).
  - Called it at the top of `vectorized_bs_price`, `vectorized_bs_delta`,
    `vectorized_bs_all_greeks` (after array coercion, before the
    deterministic mask).
  - Added `if option_type == "call" and r <= 0: return european` to
    `american_option_price`, with a comment explaining both the economics
    and the `k = 1 - exp(-r*T)` div-by-zero. Put branch untouched.
- `tests/test_option_pricer.py` — new `TestVectorizedFailLoud` class:
  6 raise tests (each of the 3 fns x {bad K, bad S}) + 3 value-equality
  tests (each fn matches the scalar pricer to `<1e-9` on an all-valid
  mixed array, all outputs finite).
- `tests/test_advanced_quant.py` — added to `TestAmericanOptionPricing`:
  `test_american_call_zero_rate_with_dividend_finite` (r=0 and r=-0.01
  calls with q>0 are finite and == European) and
  `test_american_call_positive_rate_dividend_unchanged` (r>0,q>0 BAW
  path still gives American >= European).

§2 invariant preserved by construction: the guard only RAISES on
truly-invalid inputs and changes nothing on valid inputs; no `ev_raw` /
`ev_dollars` / multiplier / decision-layer code was edited. Reviewers
stay downgrade-only.

## Evidence
<!-- Exact commands run, numbers, links to raw artifacts. -->

```text
Repro (pre-fix):
  american_option_price(100,100,0.5, r=0.0, 0.20, "call", q=0.05)  -> nan
  american_option_price(100,100,0.5, r=-0.01,0.20,"call", q=0.05)  -> 8.80 (european 4.22)
  vectorized_bs_{price,delta,all_greeks}(..., K=[..,-5,..])        -> silent NaN

Post-fix:
  american call r=0,  q>0  -> 4.419720 == european 4.419720
  american call r=-0.01,q>0 -> 4.221504 == european 4.221504
  american call r=0.05,q>0 (valid) -> 6.371015 >= european 5.498015 (BAW path intact)
  vectorized_*  bad K/S    -> ValueError "Strike/Spot price ... must be positive"

pytest tests/test_option_pricer.py tests/test_advanced_quant.py -q   -> 54 passed
pytest tests/ -q -p no:cacheprovider -m "not backtest_regression"    -> see PR (matches origin/main baseline, no NEW failures)
ruff check + ruff format --check (3 edited files)                    -> clean
```

## Unresolved / handoff
<!-- What's still open; what the next agent should look at next. -->

- **`american_option_greeks` (finite-difference Greeks)** calls
  `american_option_price` repeatedly; with the R37 fix it no longer
  risks the r=0 NaN propagating into a call's finite-difference Greeks.
  No separate guard added there — it inherits the fix.
- **NaN / non-finite S,K elements** are not explicitly handled by
  `_validate_vectorized_inputs` (it checks `<= 0`; `NaN <= 0` is False).
  Consistent with the scalar `_validate_inputs`, which also does not
  catch NaN. NaN-strike defence belongs in the data layer upstream, same
  conclusion as the S42 dossier-guards card.
- **r<0 American put** is already short-circuited to European by the
  pre-existing `put and r <= 0` block; no change needed.
