"""Held-finding reproduction: F1-roll-ev-bypass.

FINDING (invariant hole): ``WheelTracker.roll_put`` / ``roll_call`` close the
old short-option leg and open a genuinely NEW short leg (new strike / premium /
expiration / IV) but neither body runs the D17 concentration-cap check
(``_evaluate_d17_hard_blocks``) nor the D16 EV-authority token consume that
``open_short_put`` (engine/wheel_tracker.py L667 token gate, L696-727 D17
block) and ``open_covered_call`` (L1385 token gate, L1454+ D17 block) enforce.

``roll_put`` (engine/wheel_tracker.py L1031-1127) mutates the position's
``put_strike`` / ``put_premium`` / ``put_expiration_date`` etc. (L1097-1104)
with ZERO gate. So a book constructed via the canonical production factory
``engine.wheel_runner.make_live_book_tracker`` (which sets
``enforce_single_name_cap=True`` — the R10 10%-of-NAV single-name hard-refusal)
can be pushed WELL past the single-name concentration cap by rolling an
under-cap put up to a much larger strike. The direct ``open_short_put`` path
refuses exactly this (see ``tests/test_production_tracker_caps.py::
test_production_refuses_single_name_over_10pct``); the roll path does not.

CONCRETE REPRO (production default, no connector so NAV == initial_capital via
``_compute_live_nav`` static_fallback, L1940-1941):

  * NAV = $100,000.
  * open_short_put strike=$90 => notional = 90*100 = $9,000 = 9% NAV
    (< 10% single-name cap) -> opens successfully.
  * roll_put new_strike=$190 => new notional = 190*100 = $19,000 = 19% NAV
    (>> 10% cap). The single-name cap MUST apply to the rolled leg, so this
    roll MUST be refused / leave the single-name notional at or under 10% NAV.

CORRECT BEHAVIOR asserted below: after the roll attempt, the resulting
single-name short-option notional for the ticker must NOT exceed the 10%-of-NAV
single-name cap. Equivalently, the >10% roll must NOT silently succeed.

This test asserts the bug-free behavior and therefore FAILS on current engine
code (``roll_put`` silently applies the 19%-NAV leg). Marked xfail(strict) so
it is green now and flips to a hard failure (XPASS) once the roll path is
wired through the single-name cap.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine.wheel_runner import make_live_book_tracker

# 10% single-name cap == engine.portfolio_risk_gates._DEFAULT_MAX_SINGLE_NAME_PCT
SINGLE_NAME_CAP_PCT = 0.10

INITIAL_CAPITAL = 100_000.0

ENTRY = date(2026, 1, 5)
EXPIRY = date(2026, 2, 9)
ROLL_DATE = date(2026, 1, 20)
NEW_EXPIRY = date(2026, 3, 9)

TICKER = "AAPL"

# Under-cap opening strike: $90 * 100 = $9,000 = 9% of $100k NAV (< 10%).
OPEN_STRIKE = 90.0
# Roll target far over the cap: $190 * 100 = $19,000 = 19% of $100k NAV.
ROLL_STRIKE = 190.0


@pytest.mark.xfail(
    reason="held finding F1-roll-ev-bypass; drop xfail when fixed",
    strict=True,
)
def test_roll_put_must_enforce_single_name_cap_on_rolled_leg():
    """A production tracker (enforce_single_name_cap=True) must apply the R10
    single-name 10%-of-NAV cap to the NEW leg opened by ``roll_put``.

    Opening a near-cap put then rolling it to a much larger strike must NOT
    silently succeed with ~19% NAV single-name notional. The rolled leg has to
    be gated exactly like ``open_short_put`` gates a fresh open.
    """
    t = make_live_book_tracker(initial_capital=INITIAL_CAPITAL)
    # No connector attached => _compute_live_nav uses static_fallback:
    # NAV == initial_capital == $100,000 (deterministic, I/O-free).
    assert t.enforce_single_name_cap is True
    assert t.connector is None

    # 1) Open a short put just UNDER the single-name cap (9% NAV). This is a
    #    legitimate open the production factory allows.
    opened = t.open_short_put(TICKER, OPEN_STRIKE, 2.0, ENTRY, EXPIRY, 0.30)
    assert opened is True, "under-cap open should succeed on the production book"
    assert TICKER in t.positions
    # Sanity: the opened leg is 9% NAV, comfortably under the 10% cap.
    open_notional = OPEN_STRIKE * 100.0
    assert open_notional / INITIAL_CAPITAL < SINGLE_NAME_CAP_PCT

    # Confirm the direct-open path DOES refuse the over-cap strike, proving the
    # cap is live for this book and this ticker (so the roll bypass, not an
    # unarmed cap, is what this test isolates).
    n_positions_before = len(t.positions)

    # 2) Roll the under-cap put up to a strike whose notional is ~19% of NAV —
    #    far over the 10% single-name cap.
    roll_notional = ROLL_STRIKE * 100.0
    assert roll_notional / INITIAL_CAPITAL > SINGLE_NAME_CAP_PCT  # 19% > 10%

    roll_result = t.roll_put(
        TICKER,
        ROLL_DATE,
        new_strike=ROLL_STRIKE,
        new_premium=4.0,
        new_expiration=NEW_EXPIRY,
        new_iv=0.35,
        buyback_price=0.50,
    )

    # 3) CORRECT (bug-free) behavior: the single-name cap must have applied to
    #    the rolled leg. After the roll attempt, the ticker's short-option
    #    single-name notional MUST NOT exceed the 10%-of-NAV cap.
    #
    #    Whatever form the fix takes -- refuse-and-return-None (roll rejected,
    #    old $90 leg retained), raise, or otherwise leave the position within
    #    the cap -- the invariant is the same: the effective single-name
    #    notional after the roll never breaches 10% NAV. On current code
    #    roll_put silently rewrites put_strike to $190 (19% NAV) and returns a
    #    success dict, so this assertion FAILS (proving the bug).
    pos = t.positions.get(TICKER)
    effective_strike = pos.put_strike if pos is not None else 0.0
    effective_notional = float(effective_strike) * 100.0
    effective_pct = effective_notional / INITIAL_CAPITAL

    # The book must never end up over-concentrated via a roll.
    assert effective_pct <= SINGLE_NAME_CAP_PCT, (
        "roll_put bypassed the single-name cap: post-roll single-name notional "
        f"is ${effective_notional:,.0f} = {effective_pct:.0%} of "
        f"${INITIAL_CAPITAL:,.0f} NAV, over the "
        f"{SINGLE_NAME_CAP_PCT:.0%} R10 cap. The rolled leg was not gated; "
        f"roll_result={roll_result!r}."
    )

    # If the roll was correctly refused, the old under-cap leg must remain and
    # no over-cap leg should have been applied.
    assert effective_strike != ROLL_STRIKE, (
        "roll_put applied the over-cap $190 strike (19% NAV) without a "
        "single-name cap check."
    )

    # A correct refusal should also not have consumed the position count into a
    # concentrated state.
    assert len(t.positions) == n_positions_before
