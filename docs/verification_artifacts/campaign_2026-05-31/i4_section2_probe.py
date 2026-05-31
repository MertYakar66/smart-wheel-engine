"""I4 — Adversarial probe of the SACRED CLAUDE.md §2 invariant.

OBSERVE-AND-DOCUMENT ONLY. No engine/ file is modified. Read-only.

The invariant (CLAUDE.md §2): No tradeable candidate bypasses EVEngine.evaluate.
Reviewers (chart, dealer, advisor, news) can only DOWNGRADE a verdict
(proceed→review/skip→blocked), never UPGRADE. The dealer multiplier is clamped
[0.70, 1.05] and scales ev_dollars only — never ev_raw or prob_profit.
regime_multiplier is validated/clamped to [0.0, 1.25]. A genuine breach is a
negative-EV candidate becoming tradeable.

Six attacks, each tries genuinely hard to break the invariant; reports
HELD / BREACHED with computed evidence:

  1. SIGN-FLIP VIA MULTIPLIERS — sweep regime_multiplier + adversarial
     MarketStructure; verify sign(ev_dollars) == sign(ev_raw) and clamps hold.
  2. R6 + R7–R10 COMPOSITION — downgrade-only + first-match short-circuit.
  3. FORGED ev_row — dossier trusts ev_row (NOT the chokepoint); tracker
     token gate refuses the corresponding negative trade.
  4. TOKEN GATE — no token / forged token / stale-negative current_ev all refused;
     issue refuses ev<=0.
  5. R5 BOUNDARY — ev=10.0 proceed; 9.99 review.
  6. NON-FINITE EV — +inf and NaN → blocked (R1a).

Run:
  SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i4_section2_probe.py
"""
from __future__ import annotations

import math
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # load-bearing: avoid user-site .pth shadowing engine.*

# Windows console defaults to cp1252; force UTF-8 so unicode arrows in the
# diagnostics do not crash the run (and the tee'd RAW file stays UTF-8).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

import numpy as np  # noqa: E402

from datetime import datetime  # noqa: E402

from engine.ev_engine import EVEngine, ShortOptionTrade  # noqa: E402
from engine.dealer_positioning import (  # noqa: E402
    DealerAssumption,
    GammaWall,
    MarketStructure,
    dealer_regime_multiplier,
)
from engine.candidate_dossier import (  # noqa: E402
    CandidateDossier,
    EnginePhaseReviewer,
    MIN_PROCEED_EV_DOLLARS,
)
from engine.wheel_tracker import EVAuthorityRefused, WheelTracker  # noqa: E402

SEP = "=" * 88
results: dict[str, str] = {}  # attack -> HELD / BREACHED


def hdr(n: str) -> None:
    print(SEP)
    print(n)
    print(SEP)


def sign(x: float) -> int:
    if not math.isfinite(x):
        return 99
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# A fat-tail-ish negative-EV short put. Deep-ITM-ish strike above spot for a
# put (strike > spot) so it is heavily in-the-money → assignment near-certain →
# ev_raw strongly negative even before multipliers. We feed an empirical
# return distribution with a fat left tail to be adversarial about the tail
# penalty path too.
# --------------------------------------------------------------------------- #
def negative_ev_trade(regime_multiplier: float = 1.0) -> ShortOptionTrade:
    return ShortOptionTrade(
        option_type="put",
        underlying="ZZZ",
        spot=100.0,
        strike=130.0,          # deep ITM put → near-certain assignment loss
        premium=1.50,          # tiny premium vs huge intrinsic risk
        dte=35,
        iv=0.45,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        contracts=1,
        bid=1.45,
        ask=1.55,
        regime_multiplier=regime_multiplier,
    )


def fat_tail_returns(n: int = 5000, seed: int = 7) -> np.ndarray:
    """Student-t(3) log-returns scaled to ~monthly vol with a fat left tail."""
    rng = np.random.default_rng(seed)
    t = rng.standard_t(df=3, size=n) * 0.10
    return t.astype(float)


def make_market_structure(regime: str, confidence: float,
                          gex_total: float = 0.0) -> MarketStructure:
    ms = MarketStructure(
        ticker="ZZZ",
        as_of=datetime(2024, 7, 8),
        spot=100.0,
        expiry=date(2024, 8, 16),
        assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
        gex_total=gex_total,
    )
    ms.regime = regime  # type: ignore[assignment]
    ms.confidence = confidence
    return ms


# =========================================================================== #
# ATTACK 1 — SIGN-FLIP VIA MULTIPLIERS
# =========================================================================== #
def attack_1() -> None:
    hdr("ATTACK 1 — SIGN-FLIP VIA MULTIPLIERS (ev_engine.py:498-538)")
    eng = EVEngine()
    fwd = fat_tail_returns()

    # Baseline ev_raw (regime_multiplier=1.0, no market_structure). ev_raw is
    # invariant to the multiplier so we read mean_pnl as ev_raw.
    base = eng.evaluate(negative_ev_trade(1.0), forward_log_returns=fwd)
    ev_raw = base.mean_pnl
    print(f"baseline ev_raw (mean_pnl) = {ev_raw:.4f}   prob_profit = {base.prob_profit:.4f}")
    print(f"baseline heavy_tail = {base.heavy_tail}  tail_xi = {base.tail_xi}")
    assert ev_raw < 0, "test setup failure: trade is not negative-EV"

    print()
    print("(a) Sweep trade.regime_multiplier, NO market_structure:")
    print(f"  {'regime_in':>12} {'regime_mult':>12} {'dealer_mult':>12} "
          f"{'ev_raw':>14} {'ev_dollars':>14} {'sign_match':>10} {'clamp_ok':>9}")
    sweep = [-100.0, -1.0, 0.0, 0.5, 1.0, 1.25, 5.0, 1e9, float("inf"), float("nan")]
    ok_all = True
    for rm in sweep:
        r = eng.evaluate(negative_ev_trade(rm), forward_log_returns=fwd)
        # ev_raw must be unchanged by the multiplier (it is mean_pnl).
        raw = r.mean_pnl
        clamp_ok = (0.0 <= r.regime_multiplier <= 1.25 * 1.0 + 1e-9)  # heavy-tail penalty halves
        # account for heavy_tail penalty (0.5) AND dealer (none here): effective
        # cap is 1.25 * heavy_tail_penalty. Recompute the true cap:
        cap = 1.25 * (0.5 if r.heavy_tail else 1.0)
        clamp_ok = (-1e-9 <= r.regime_multiplier <= cap + 1e-9)
        # sign(ev_dollars) must equal sign(ev_raw) UNLESS regime_mult==0 (→ 0).
        if r.regime_multiplier == 0.0:
            sign_match = (r.ev_dollars == 0.0)
        else:
            sign_match = (sign(r.ev_dollars) == sign(raw))
        ok_all = ok_all and sign_match and clamp_ok
        anomaly = r.metadata.get("regime_anomaly", "")
        print(f"  {rm:>12.4g} {r.regime_multiplier:>12.5f} {r.dealer_multiplier:>12.5f} "
              f"{raw:>14.4f} {r.ev_dollars:>14.4f} {str(sign_match):>10} {str(clamp_ok):>9}"
              + (f"   anomaly={anomaly}" if anomaly else ""))

    print()
    print("(b) Adversarial MarketStructure (try to exceed 1.05 or go negative):")
    print(f"  {'regime':>26} {'conf':>6} {'dealer_mult':>12} {'in[0.70,1.05]?':>14} "
          f"{'ev_dollars':>14} {'sign_match':>10}")
    # Standard regimes at extreme confidence + an out-of-range confidence to try
    # to push the multiplier past its clamp.
    ms_cases = [
        ("long_gamma_dampening", 1.0),
        ("long_gamma_dampening", 5.0),     # conf > 1 → clamped to 1 inside fn
        ("long_gamma_dampening", -3.0),    # conf < 0 → clamped to 0
        ("short_gamma_amplifying", 1.0),
        ("short_gamma_amplifying", 100.0),
        ("near_flip", 1.0),
        ("neutral", 1.0),
    ]
    for regime, conf in ms_cases:
        ms = make_market_structure(regime, conf)
        dm = dealer_regime_multiplier(ms)
        in_range = (0.70 - 1e-9 <= dm <= 1.05 + 1e-9)
        r = eng.evaluate(negative_ev_trade(1.0), forward_log_returns=fwd, market_structure=ms)
        if r.regime_multiplier == 0.0:
            sm = (r.ev_dollars == 0.0)
        else:
            sm = (sign(r.ev_dollars) == sign(r.mean_pnl))
        ok_all = ok_all and in_range and sm
        print(f"  {regime:>26} {conf:>6.1f} {dm:>12.5f} {str(in_range):>14} "
              f"{r.ev_dollars:>14.4f} {str(sm):>10}")

    # (c) Try a forged MarketStructure-like object that returns a >1.05 / negative
    # multiplier through dealer_regime_multiplier — the only way is to bypass the
    # function, which evaluate() does NOT permit (it always calls the fn).
    print()
    print("(c) Forged regime label 'totally_made_up' → falls through to 1.0:")
    ms_bad = make_market_structure("totally_made_up", 1.0)  # type: ignore[arg-type]
    dm_bad = dealer_regime_multiplier(ms_bad)
    print(f"  dealer_regime_multiplier(forged regime) = {dm_bad}  (fall-through 1.0, no breach)")
    ok_all = ok_all and (dm_bad == 1.0)

    # (d) Positive-EV sanity: confirm a long-gamma boost can RAISE a positive EV
    # (allowed — that is not a breach; breach is only sign-flip of a NEGATIVE EV).
    pos = ShortOptionTrade(option_type="put", underlying="QQQ", spot=100.0,
                           strike=85.0, premium=2.0, dte=30, iv=0.30,
                           bid=1.95, ask=2.05)
    pos_fwd = (np.random.default_rng(1).standard_normal(5000) * 0.05).astype(float)
    pr = eng.evaluate(pos, forward_log_returns=pos_fwd)
    ms_boost = make_market_structure("long_gamma_dampening", 1.0)
    pr_boost = eng.evaluate(pos, forward_log_returns=pos_fwd, market_structure=ms_boost)
    print()
    print("(d) Positive-EV: long-gamma boost scales UP a POSITIVE ev "
          "(allowed; not a breach):")
    print(f"  ev_dollars no-MS={pr.ev_dollars:.4f}  with 1.05 boost={pr_boost.ev_dollars:.4f}  "
          f"ratio={pr_boost.ev_dollars / pr.ev_dollars:.5f}")
    # The boost must not exceed 1.05x and the underlying sign stays positive.
    boost_ok = pr.ev_dollars > 0 and pr_boost.ev_dollars > 0 and \
        pr_boost.ev_dollars <= pr.ev_dollars * 1.05 + 1e-6
    ok_all = ok_all and boost_ok

    results["ATTACK 1 (sign-flip via multipliers)"] = "HELD" if ok_all else "BREACHED"
    print(f"\n  >>> ATTACK 1: {results['ATTACK 1 (sign-flip via multipliers)']}")


# =========================================================================== #
# ATTACK 2 — R6 + R7–R10 COMPOSITION (downgrade-only, first-match)
# =========================================================================== #
def _ok_chart():
    """A minimal chart context that is_ok() (non-None screenshot_path, no error)
    and agrees on the engine spot of 100.0 so R3 does not fire."""
    from pathlib import Path

    from engine.chart_context import ChartContext
    return ChartContext(
        ticker="ZZZ",
        timeframe="1D",
        captured_at=datetime(2024, 7, 8),
        screenshot_path=Path("nonexistent_probe.png"),  # non-None → is_ok() True
        visible_price=100.0,
        visible_indicators={},
        source="probe",
        error="",
    )


def attack_2() -> None:
    hdr("ATTACK 2 — R6 + R7-R10 COMPOSITION (candidate_dossier.py:319-495)")
    from engine.portfolio_risk_gates import PortfolioContext

    reviewer = EnginePhaseReviewer()

    # ev_row positive enough to reach R5 proceed (>=10), so R6-R10 are reachable.
    base_row = {
        "ticker": "ZZZ", "spot": 100.0, "strike": 95.0, "premium": 2.0,
        "dte": 35, "iv": 0.45, "ev_dollars": 50.0, "prob_profit": 0.65,
        "contracts": 1,
    }

    # (1) R6: short-gamma + strike at/above put wall → review.
    ms = make_market_structure("short_gamma_amplifying", 1.0, gex_total=-1e9)
    ms.nearest_put_wall = GammaWall(strike=94.0, distance_pct=-0.06,
                                    net_gex=-5e8, side="put")
    d_r6 = CandidateDossier(ticker="ZZZ", ev_row=dict(base_row),
                            chart_context=_ok_chart(), market_structure=ms)
    v6, r6, _ = reviewer.review(d_r6)
    print(f"  R6 alone (short-gamma + strike95 >= put-wall94): verdict={v6} reason={r6}")

    # (2) Build a portfolio_context that should also fire a downgrade (R7/R8/R9/R10).
    # Use a deliberately tiny NAV so the single-name / sector notional caps blow.
    pc = PortfolioContext(
        nav=10_000.0,                       # 95*100 = 9500 notional ≈ 95% of NAV
        held_option_positions=[],
        spot_prices={"ZZZ": 100.0},
    )
    d_both = CandidateDossier(ticker="ZZZ", ev_row=dict(base_row),
                              chart_context=_ok_chart(), market_structure=ms,
                              portfolio_context=pc)
    v_both, r_both, notes_both = reviewer.review(d_both)
    print(f"  R6 + portfolio_ctx BOTH attached: verdict={v_both} reason={r_both}  "
          f"(first-match short-circuit)")
    for n in notes_both:
        print(f"      note: {n}")

    # (3) portfolio_context only (no market_structure) so R7-R10 are the first
    # downgrade after R5 proceed. Confirm one of them fires and it is a DOWNGRADE.
    d_pc = CandidateDossier(ticker="ZZZ", ev_row=dict(base_row),
                            chart_context=_ok_chart(), portfolio_context=pc)
    v_pc, r_pc, notes_pc = reviewer.review(d_pc)
    print(f"  portfolio_ctx only (tiny NAV): verdict={v_pc} reason={r_pc}")
    for n in notes_pc:
        print(f"      note: {n}")

    # (4) Negative EV with BOTH downgraders attached → R1 still wins (blocked),
    # downgraders never get a chance to "rescue".
    neg_row = dict(base_row); neg_row["ev_dollars"] = -25.0
    d_neg = CandidateDossier(ticker="ZZZ", ev_row=neg_row,
                             chart_context=_ok_chart(), market_structure=ms,
                             portfolio_context=pc)
    v_neg, r_neg, _ = reviewer.review(d_neg)
    print(f"  NEGATIVE ev + R6 + portfolio_ctx: verdict={v_neg} reason={r_neg}  "
          "(R1 short-circuits FIRST; downgraders cannot rescue)")

    # Verdicts: every reachable rule must be a downgrade from 'proceed' (or stay
    # 'blocked' for negative). None may produce a verdict 'stronger' than proceed.
    downgrade_only = (
        v6 in ("review", "skip", "blocked")
        and v_both in ("review", "skip", "blocked")
        and v_pc in ("review", "skip", "blocked")
        and v_neg == "blocked" and r_neg == "negative_ev"
    )
    # First-match: with both attached, R6 (dealer) is evaluated before R7-R10, so
    # the reason should be the R6 reason when R6 fires.
    first_match_ok = (r_both == r6)
    held = downgrade_only and first_match_ok
    results["ATTACK 2 (R6+R7-R10 composition)"] = "HELD" if held else "BREACHED"
    print(f"\n  downgrade_only={downgrade_only}  first_match(R6 before R7-R10)={first_match_ok}")
    print(f"  >>> ATTACK 2: {results['ATTACK 2 (R6+R7-R10 composition)']}")


# =========================================================================== #
# ATTACK 3 — FORGED ev_row passes dossier; tracker token gate refuses
# =========================================================================== #
def attack_3() -> None:
    hdr("ATTACK 3 — FORGED ev_row: dossier is NOT the chokepoint; tracker gate is")
    reviewer = EnginePhaseReviewer()

    # A hand-crafted ev_row claiming a fat positive EV but whose UNDERLYING trade
    # (deep-ITM put, premium tiny) is genuinely negative-EV. The dossier trusts
    # ev_row['ev_dollars'] verbatim (candidate_dossier.py:88-89) → proceeds.
    forged_row = {
        "ticker": "ZZZ", "spot": 100.0, "strike": 130.0, "premium": 1.50,
        "dte": 35, "iv": 0.45, "ev_dollars": 999.0,  # FORGED positive
        "prob_profit": 0.95, "contracts": 1, "distribution_source": "forged",
    }
    d = CandidateDossier(ticker="ZZZ", ev_row=forged_row, chart_context=_ok_chart())
    v, r, _ = reviewer.review(d)
    print(f"  Forged ev_dollars=999 in dossier: verdict={v} reason={r}  "
          "(dossier trusts ev_row — by design, NOT the chokepoint)")
    dossier_trusts = (v == "proceed")

    # Now compute the REAL EV of that same trade via the engine.
    eng = EVEngine()
    real = eng.evaluate(negative_ev_trade(1.0), forward_log_returns=fat_tail_returns())
    print(f"  REAL engine ev_dollars for the same trade = {real.ev_dollars:.2f} "
          f"(prob_profit={real.prob_profit:.3f}) → genuinely negative-EV")

    # The §2 chokepoint: tracker token gate. Issuing a token from the REAL
    # (negative) row is refused; consuming with the real negative ev is refused.
    tr = WheelTracker(initial_capital=1_000_000.0, require_ev_authority=True)
    real_row = dict(forged_row); real_row["ev_dollars"] = real.ev_dollars
    issue_refused = False
    try:
        tr.issue_ev_authority_token(real_row)
    except EVAuthorityRefused as ex:
        issue_refused = True
        print(f"  tracker.issue_ev_authority_token(real negative row) → REFUSED: {ex}")

    # Even if an attacker forges the row to issue a token (ev_dollars=999), the
    # consume re-check with the REAL current ev refuses.
    forged_token = tr.issue_ev_authority_token(forged_row)  # issues (999>0)
    opened = tr.open_short_put(
        ticker="ZZZ", strike=130.0, premium=1.50,
        entry_date=date(2024, 7, 8), expiration_date=date(2024, 8, 12),
        iv=0.45, ev_authority_token=forged_token,
        current_ev_dollars=real.ev_dollars,   # REAL negative EV at fire time
    )
    print(f"  open_short_put(forged token, current_ev={real.ev_dollars:.2f}<0) → "
          f"opened={opened}  (stale_ev consume reject)")
    consume_refused = (opened is False)

    held = dossier_trusts and issue_refused and consume_refused and real.ev_dollars < 0
    results["ATTACK 3 (forged ev_row / chokepoint)"] = "HELD" if held else "BREACHED"
    print(f"\n  dossier_trusts_forged={dossier_trusts}  issue_refused={issue_refused}  "
          f"consume_refused_on_real_negative={consume_refused}")
    print(f"  >>> ATTACK 3: {results['ATTACK 3 (forged ev_row / chokepoint)']}")


# =========================================================================== #
# ATTACK 4 — TOKEN GATE (require_ev_authority=True)
# =========================================================================== #
def attack_4() -> None:
    hdr("ATTACK 4 — TOKEN GATE (wheel_tracker.py:399-516)")
    # Large NAV so the D17 downstream caps (delta/sector/single-name/Kelly) do
    # NOT mask the token-gate outcome we are probing. (At $1M NAV the 300-delta
    # /$100k portfolio-delta cap = $3000 delta-$ budget, and a single ~ATM short
    # put already carries ~$4240 delta-$, so a smaller NAV would refuse the
    # happy path on the *delta* gate — a downstream safety gate, NOT a §2 issue.
    # We isolate the §2 token gate by giving the delta gate ample headroom.)
    tr = WheelTracker(initial_capital=100_000_000.0, require_ev_authority=True)
    common = dict(ticker="AAA", strike=90.0, premium=2.0,
                  entry_date=date(2024, 7, 8), expiration_date=date(2024, 8, 12),
                  iv=0.30)

    # (a) No token at all.
    a = tr.open_short_put(**common, ev_authority_token=None, current_ev_dollars=50.0)
    print(f"  (a) no token                          → opened={a} (expect False)")

    # (b) Random / forged token string.
    b = tr.open_short_put(**common, ev_authority_token="deadbeef" * 8,
                          current_ev_dollars=50.0)
    print(f"  (b) forged token string               → opened={b} (expect False)")

    # (c) Valid token issued for a positive-EV row, consumed with a stale/negative
    # current_ev_dollars → stale_ev reject.
    pos_row = {"ticker": "AAA", "strike": 90.0, "premium": 2.0, "dte": 35,
               "ev_dollars": 50.0, "prob_profit": 0.7, "distribution_source": "x"}
    tok = tr.issue_ev_authority_token(pos_row)
    c = tr.open_short_put(**common, ev_authority_token=tok, current_ev_dollars=-5.0)
    print(f"  (c) valid token + current_ev=-5.0     → opened={c} (expect False, stale_ev)")
    # token retained on stale reject — confirm it is still in the set.
    tok_retained = tok in tr._ev_authority_tokens
    print(f"      token retained after stale reject = {tok_retained} (expect True)")

    # (c2) missing current_ev_dollars (None) in strict mode → reject.
    tok2 = tr.issue_ev_authority_token({**pos_row, "strike": 91.0})
    common2 = {**common, "strike": 91.0}
    c2 = tr.open_short_put(**common2, ev_authority_token=tok2, current_ev_dollars=None)
    print(f"  (c2) valid token + current_ev=None    → opened={c2} (expect False, missing)")

    # (d) issue refuses non-positive ev_dollars.
    neg_issue_refused = False
    try:
        tr.issue_ev_authority_token({**pos_row, "ev_dollars": -0.01})
    except EVAuthorityRefused as ex:
        neg_issue_refused = True
        print(f"  (d) issue(ev_dollars=-0.01)           → REFUSED: {type(ex).__name__}")
    zero_issue_refused = False
    try:
        tr.issue_ev_authority_token({**pos_row, "ev_dollars": 0.0})
    except EVAuthorityRefused:
        zero_issue_refused = True
        print("  (d) issue(ev_dollars=0.0)             → REFUSED (<=0 boundary)")

    # (e) the happy path: valid token + positive current_ev → opens (proof the
    # gate is not just refusing everything).
    tok3 = tr.issue_ev_authority_token({**pos_row, "strike": 92.0})
    common3 = {**common, "strike": 92.0}
    e = tr.open_short_put(**common3, ev_authority_token=tok3, current_ev_dollars=50.0)
    print(f"  (e) valid token + current_ev=50.0     → opened={e} (expect True; gate not vacuous)")
    # single-use: replaying the consumed token fails.
    e2 = tr.open_short_put(**{**common, "strike": 93.0}, ev_authority_token=tok3,
                           current_ev_dollars=50.0)
    print(f"  (e2) replay consumed token            → opened={e2} (expect False, single-use)")

    held = (a is False and b is False and c is False and tok_retained
            and c2 is False and neg_issue_refused and zero_issue_refused
            and e is True and e2 is False)
    results["ATTACK 4 (token gate)"] = "HELD" if held else "BREACHED"
    print(f"\n  >>> ATTACK 4: {results['ATTACK 4 (token gate)']}")


# =========================================================================== #
# ATTACK 5 — R5 BOUNDARY (ev=10.0 proceed; 9.99 review)
# =========================================================================== #
def attack_5() -> None:
    hdr("ATTACK 5 — R5 BOUNDARY (candidate_dossier.py:310, MIN_PROCEED_EV="
        f"{MIN_PROCEED_EV_DOLLARS})")
    reviewer = EnginePhaseReviewer()
    row = {"ticker": "ZZZ", "spot": 100.0, "strike": 95.0, "premium": 2.0,
           "dte": 35, "iv": 0.45, "contracts": 1}

    def verdict_at(ev: float):
        d = CandidateDossier(ticker="ZZZ", ev_row={**row, "ev_dollars": ev},
                             chart_context=_ok_chart())
        return reviewer.review(d)

    v_10, r_10, _ = verdict_at(10.0)
    v_999, r_999, _ = verdict_at(9.99)
    v_1001, r_1001, _ = verdict_at(10.01)
    print(f"  ev=10.00 → verdict={v_10} reason={r_10} (expect proceed)")
    print(f"  ev= 9.99 → verdict={v_999} reason={r_999} (expect review)")
    print(f"  ev=10.01 → verdict={v_1001} reason={r_1001} (expect proceed)")
    held = (v_10 == "proceed" and v_999 == "review" and v_1001 == "proceed")
    results["ATTACK 5 (R5 boundary)"] = "HELD" if held else "BREACHED"
    print(f"\n  >>> ATTACK 5: {results['ATTACK 5 (R5 boundary)']}")


# =========================================================================== #
# ATTACK 6 — NON-FINITE EV → blocked (R1a)
# =========================================================================== #
def attack_6() -> None:
    hdr("ATTACK 6 — NON-FINITE EV (candidate_dossier.py:266-268, R1a)")
    reviewer = EnginePhaseReviewer()
    row = {"ticker": "ZZZ", "spot": 100.0, "strike": 95.0, "premium": 2.0,
           "dte": 35, "iv": 0.45, "contracts": 1}

    def verdict_at(ev: float):
        d = CandidateDossier(ticker="ZZZ", ev_row={**row, "ev_dollars": ev},
                             chart_context=_ok_chart())
        return reviewer.review(d)

    v_pinf, r_pinf, _ = verdict_at(float("inf"))
    v_ninf, r_ninf, _ = verdict_at(float("-inf"))
    v_nan, r_nan, _ = verdict_at(float("nan"))
    print(f"  ev=+inf → verdict={v_pinf} reason={r_pinf} (expect blocked / ev_non_finite)")
    print(f"  ev=-inf → verdict={v_ninf} reason={r_ninf} (expect blocked / ev_non_finite)")
    print(f"  ev= nan → verdict={v_nan} reason={r_nan} (expect blocked / ev_non_finite)")
    held = (v_pinf == "blocked" and r_pinf == "ev_non_finite"
            and v_ninf == "blocked" and r_ninf == "ev_non_finite"
            and v_nan == "blocked" and r_nan == "ev_non_finite")
    results["ATTACK 6 (non-finite EV)"] = "HELD" if held else "BREACHED"
    print(f"\n  >>> ATTACK 6: {results['ATTACK 6 (non-finite EV)']}")


def main() -> int:
    print(f"PROVIDER={os.environ.get('SWE_DATA_PROVIDER', '(unset)')}")
    print(f"ROOT={ROOT}")
    print(f"engine.ev_engine from: {EVEngine.__module__} "
          f"({sys.modules['engine.ev_engine'].__file__})")
    print()
    attack_1()
    attack_2()
    attack_3()
    attack_4()
    attack_5()
    attack_6()

    hdr("SUMMARY")
    breach = False
    for k, v in results.items():
        print(f"  {v:>9}  {k}")
        if v != "HELD":
            breach = True
    print()
    print("  §2 INVARIANT:", "BREACHED — SEE ABOVE" if breach else
          "HELD across all 6 attacks — could not break it")
    return 1 if breach else 0


if __name__ == "__main__":
    raise SystemExit(main())
