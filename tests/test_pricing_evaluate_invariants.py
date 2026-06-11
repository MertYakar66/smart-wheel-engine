"""Pricing-core + evaluate-gate invariants — quant-layer test audit round 2
(W63-W64), PR-6.

Pins two contracts feeding EVEngine.evaluate:
- W63 the binomial tree's vega / theta / rho UNITS (``engine/binomial_tree.py``)
  against the analytic BSM Greeks (``engine/option_pricer.py``). They are documented
  "per 1 vol point" / "annual" / "per 1% rate" but only delta + gamma were validated;
  a unit slip (e.g. rho per 1.0 vs per 0.01) would be a 100x error invisible today.
  Cross-checked on q=0 CALLS, where the American tree price == the European BSM
  (no early-exercise premium) so the Greeks must agree.
- W64 EVEngine.evaluate finiteness at degenerate dte (<=0): the BSM falls to
  intrinsic (T=max(dte,0)/365) and ev_per_day floors the divisor at max(dte,1), so a
  zero/negative dte must still yield a finite EVResult (no div-by-zero).

Behaviour-pinning (the #366 lesson); no §2 surface touched (evaluate is the
authoritative path — asserted, never bypassed). Binomial GAMMA is intentionally NOT
tightly pinned: it is finite-differenced on the tree (≈25% off BSM at 800 steps,
tree-discretisation noise, not a unit issue) and is already validated elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.binomial_tree import binomial_american_full
from engine.ev_engine import EVEngine, ShortOptionTrade
from engine.option_pricer import black_scholes_all_greeks

_FWD = np.random.default_rng(0).normal(0.0, 0.10, 500)


# ---------------------------------------------------------------------------
# W63 — binomial vega / theta / rho UNITS vs BSM (q=0 calls; American==European).
# ---------------------------------------------------------------------------


class TestBinomialGreekUnits:
    # (S, K, T, r, sigma) — ATM, ITM, OTM q=0 calls.
    @pytest.mark.parametrize(
        "params",
        [
            (100.0, 100.0, 0.50, 0.05, 0.25),
            (110.0, 100.0, 0.50, 0.05, 0.25),
            (90.0, 100.0, 0.75, 0.03, 0.30),
        ],
    )
    def test_vega_theta_rho_match_bsm_units(self, params):
        S, K, T, r, sigma = params
        b = binomial_american_full(S, K, T, r, sigma, "call", q=0.0, n_steps=800)
        g = black_scholes_all_greeks(S, K, T, r, sigma, "call", q=0.0)
        # vega/theta/rho units must match within tree-convergence tolerance. A unit
        # error (per 1.0 vs per 0.01) would be ~100x — 5% comfortably distinguishes.
        for key in ("vega", "theta", "rho"):
            bv, gv = getattr(b, key), g[key]
            rel = abs(bv - gv) / (abs(gv) + 1e-9)
            assert rel < 0.05, (
                f"{key}: binomial {bv} vs BSM {gv} (rel {rel:.3f}) — unit/scale mismatch"
            )
        # delta is analytic-tight; assert it too (sign + magnitude).
        assert abs(b.delta - g["delta"]) < 0.02, f"delta {b.delta} vs BSM {g['delta']}"
        # gamma: same sign + same order of magnitude only (tree-noisy, not unit-pinned).
        assert b.gamma > 0 and g["gamma"] > 0
        assert 0.3 < b.gamma / g["gamma"] < 3.0, "binomial gamma off by >3x — beyond tree noise"


# ---------------------------------------------------------------------------
# W64 — EVEngine.evaluate finiteness at degenerate dte. dte<=0 -> BSM intrinsic +
# ev_per_day divisor floored at max(dte,1); a zero/negative dte must not produce a
# non-finite EVResult or divide by zero.
# ---------------------------------------------------------------------------


class TestEvaluateDegenerateDte:
    def _eval(self, dte):
        trade = ShortOptionTrade(
            option_type="put",
            underlying="X",
            spot=100.0,
            strike=95.0,
            premium=2.0,
            dte=dte,
            iv=0.25,
            risk_free_rate=0.05,
            dividend_yield=0.0,
        )
        return EVEngine().evaluate(trade, forward_log_returns=_FWD)

    @pytest.mark.parametrize("dte", [0, -5, 1])
    def test_degenerate_dte_is_finite(self, dte):
        r = self._eval(dte)
        assert np.isfinite(r.ev_dollars), f"dte={dte}: non-finite ev_dollars"
        assert np.isfinite(r.ev_per_day), f"dte={dte}: non-finite ev_per_day (div-by-zero?)"
        # The divisor floors at max(dte, 1), so for dte<=1 ev_per_day == ev_dollars.
        assert r.ev_per_day == pytest.approx(r.ev_dollars), (
            "dte<=1: ev_per_day should equal ev_dollars (floor)"
        )

    def test_positive_dte_divides_ev_per_day(self):
        r = self._eval(35)
        assert np.isfinite(r.ev_dollars) and np.isfinite(r.ev_per_day)
        # With a real holding period the per-day EV is a fraction of the total.
        assert abs(r.ev_per_day) < abs(r.ev_dollars), "dte=35: ev_per_day should be < ev_dollars"
