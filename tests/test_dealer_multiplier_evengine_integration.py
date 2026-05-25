"""Dealer-multiplier integration boundary tests through ``EVEngine.evaluate``.

The dealer regime multiplier (``engine.dealer_positioning.dealer_regime_multiplier``)
is one of CLAUDE.md §2's three named invariants: it must stay clamped to
``[0.70, 1.05]``, must be asymmetric by design (long-gamma boost capped at
+0.05 vs short-gamma cut down to −0.30), and must scale ``ev_dollars`` only
(never rescue a negative-EV trade or touch anything else).

Existing coverage in ``tests/test_dealer_positioning.py``:

* ``TestMultiplierBounds`` (5 tests) pins the helper-function values at all
  four regimes — but only at the unit level, never through
  ``EVEngine.evaluate``.
* ``TestEVEngineIntegration::test_short_gamma_structure_shrinks_ev`` pins
  ordering (``r_long.ev_dollars > r_short.ev_dollars``) — not the exact
  boundary values that would survive a regression applying the multiplier
  twice or capping at a wrong value.
* ``TestEVEngineIntegration::test_dealer_mult_cannot_rescue_negative_ev``
  covers the negative-EV-with-favorable-multiplier case.

What this file pins (the gap):

1. **Boundary values survive the integration pipeline** to the
   ``EVResult.dealer_multiplier`` field — exact 1.05 / 0.70 / 0.85 / 1.00.
2. **The asymmetric ``[0.70, 1.05]`` clamp** holds at the EVResult level
   (``|1.0 − 0.70| > |1.05 − 1.0|``).
3. **The ``regime_multiplier *= dealer_mult`` compounding** at
   ``engine/ev_engine.py:488`` actually fires and produces the right
   product on the ``EVResult.regime_multiplier`` field.
4. **The §2 "scales ev_dollars only" claim** expressed as proportionality:
   two evaluate calls with identical inputs except for ``MarketStructure``
   produce ``ev_dollars`` whose ratio matches the multiplier ratio. (``ev_raw``
   is a local variable in ``EVEngine.evaluate`` at line 366 — not an
   ``EVResult`` field — so the un-multiplied value can't be read directly.
   Proportionality is the correct expression of the invariant given the
   actual schema.)

Companion to PR #185's test #10 (``test_blocked_path_does_not_apply_dealer_multiplier``),
which pinned the complementary case: blocked candidates short-circuit BEFORE
the dealer math runs.

``forward_log_returns`` uses ``n=199`` (below the EVT-fit threshold of 200)
to deterministically keep ``heavy_tail=False`` so the
``regime_mult *= self.heavy_tail_penalty`` line at
``engine/ev_engine.py:474`` does not confound the multiplier-compounding
assertions in tests 6 and 7.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from engine.dealer_positioning import DealerAssumption, MarketStructure
from engine.ev_engine import EVEngine, ShortOptionTrade


# ----------------------------------------------------------------------
# Shared real-class fixtures
# ----------------------------------------------------------------------
def _trade(regime_multiplier: float = 1.0) -> ShortOptionTrade:
    """Real-shape ShortOptionTrade with positive-EV defaults.

    Mirrors ``tests/test_dealer_positioning.py::TestEVEngineIntegration._trade()``
    so the post-fix EVResult shape stays comparable.
    """
    return ShortOptionTrade(
        option_type="put",
        underlying="TEST",
        spot=100.0,
        strike=95.0,
        premium=1.20,
        bid=1.15,
        ask=1.25,
        dte=30,
        iv=0.22,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        contracts=1,
        open_interest=1000,
        regime_multiplier=regime_multiplier,
    )


def _market_structure(regime: str, confidence: float = 1.0) -> MarketStructure:
    """Minimal real MarketStructure with the requested regime + confidence."""
    return MarketStructure(
        ticker="TEST",
        as_of=datetime.now(UTC),
        spot=100.0,
        expiry=date.today() + timedelta(days=30),
        assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
        regime=regime,
        confidence=confidence,
    )


def _forward_returns(seed: int = 1) -> np.ndarray:
    """Deterministic synthetic returns with ``n=199`` (below EVT threshold).

    Keeping ``n < 200`` skips the GPD tail fit so ``heavy_tail`` stays
    False; this prevents the ``regime_mult *= self.heavy_tail_penalty``
    line at ``engine/ev_engine.py:474`` from confounding the
    multiplier-compounding assertions in tests 6 and 7.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, 0.012, 199)


# ======================================================================
# 1-4. Per-regime exact boundary values on EVResult.dealer_multiplier
# ======================================================================
class TestDealerMultiplierBoundsAtEVResultLevel:
    """The four regime → multiplier mappings survive ``EVEngine.evaluate``
    and arrive at the ``EVResult.dealer_multiplier`` field as the exact
    helper-function values.

    ``TestMultiplierBounds`` in ``tests/test_dealer_positioning.py`` pins
    these at the helper-function unit level; this class pins them after
    integration through ``EVEngine.evaluate``. A regression that applied
    the multiplier twice, or hard-coded an EVResult value, would pass
    the helper unit tests and fail here.
    """

    def test_long_gamma_full_confidence_yields_dealer_multiplier_105(self):
        ms = _market_structure("long_gamma_dampening", confidence=1.0)
        result = EVEngine().evaluate(
            _trade(), forward_log_returns=_forward_returns(), market_structure=ms
        )
        assert result.dealer_multiplier == pytest.approx(1.05, abs=1e-6)
        assert result.dealer_regime == "long_gamma_dampening"

    def test_short_gamma_full_confidence_yields_dealer_multiplier_070(self):
        ms = _market_structure("short_gamma_amplifying", confidence=1.0)
        result = EVEngine().evaluate(
            _trade(), forward_log_returns=_forward_returns(), market_structure=ms
        )
        assert result.dealer_multiplier == pytest.approx(0.70, abs=1e-6)
        assert result.dealer_regime == "short_gamma_amplifying"

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_near_flip_yields_dealer_multiplier_085_regardless_of_confidence(
        self, confidence: float
    ):
        """``near_flip`` is flat 0.85 — independent of confidence per the
        helper at ``engine/dealer_positioning.py:743-745`` ("near-flip is
        a warning regardless of confidence")."""
        ms = _market_structure("near_flip", confidence=confidence)
        result = EVEngine().evaluate(
            _trade(), forward_log_returns=_forward_returns(), market_structure=ms
        )
        assert result.dealer_multiplier == pytest.approx(0.85, abs=1e-6)
        assert result.dealer_regime == "near_flip"

    def test_neutral_regime_yields_dealer_multiplier_100(self):
        """``regime='neutral'`` with an attached MarketStructure (different
        code path from ``market_structure=None``) → identity multiplier."""
        ms = _market_structure("neutral", confidence=1.0)
        result = EVEngine().evaluate(
            _trade(), forward_log_returns=_forward_returns(), market_structure=ms
        )
        assert result.dealer_multiplier == pytest.approx(1.00, abs=1e-6)
        assert result.dealer_regime == "neutral"


# ======================================================================
# 5. §2 asymmetric-by-design at the EVResult level
# ======================================================================
class TestDealerMultiplierAsymmetricClampAtEVResultLevel:
    """Pins CLAUDE.md §2's "asymmetric by design" wording at the
    integration level: the down-cut is larger in magnitude than the
    up-boost, so a regression that accidentally symmetrised the clamp
    (e.g. to ``[0.95, 1.05]``) would be caught here."""

    def test_clamp_is_asymmetric_short_cut_larger_than_long_boost(self):
        r_long = EVEngine().evaluate(
            _trade(),
            forward_log_returns=_forward_returns(),
            market_structure=_market_structure("long_gamma_dampening", 1.0),
        )
        r_short = EVEngine().evaluate(
            _trade(),
            forward_log_returns=_forward_returns(),
            market_structure=_market_structure("short_gamma_amplifying", 1.0),
        )
        long_boost = r_long.dealer_multiplier - 1.0  # +0.05
        short_cut = 1.0 - r_short.dealer_multiplier  # +0.30
        assert short_cut > long_boost, (
            f"asymmetric-by-design broken: short_cut={short_cut:.4f} "
            f"vs long_boost={long_boost:.4f}; CLAUDE.md §2 says short-gamma "
            f"cut must be larger in magnitude than long-gamma boost"
        )
        # Pin the exact 6× ratio so a partial loosening (e.g. ±0.05
        # symmetric) doesn't sneak through.
        assert short_cut == pytest.approx(0.30, abs=1e-6)
        assert long_boost == pytest.approx(0.05, abs=1e-6)


# ======================================================================
# 6. Compounding — trade.regime_multiplier × dealer_mult on EVResult
# ======================================================================
class TestRegimeMultiplierCompounding:
    """The line ``regime_mult *= dealer_mult`` at ``engine/ev_engine.py:488``
    has no existing test. This class pins that
    ``EVResult.regime_multiplier`` actually contains the product of
    ``trade.regime_multiplier`` and ``dealer_regime_multiplier(ms)``."""

    def test_regime_multiplier_field_equals_trade_regime_times_dealer_multiplier(
        self,
    ):
        """trade.regime_multiplier=1.20 × dealer=1.05 → 1.26 on EVResult."""
        ms = _market_structure("long_gamma_dampening", 1.0)
        result = EVEngine().evaluate(
            _trade(regime_multiplier=1.20),
            forward_log_returns=_forward_returns(),
            market_structure=ms,
        )
        # n=199 returns keeps heavy_tail=False so the heavy_tail_penalty
        # at line 474 doesn't apply.
        assert result.heavy_tail is False, (
            "heavy_tail should not fire with n=199 returns; "
            "if it does, the compounding test is confounded"
        )
        # 1.20 × 1.05 = 1.26 (the dealer_mult multiplies the
        # already-set trade.regime_multiplier).
        assert result.regime_multiplier == pytest.approx(1.20 * 1.05, abs=1e-6)
        # Dealer multiplier is independently visible on EVResult.
        assert result.dealer_multiplier == pytest.approx(1.05, abs=1e-6)


# ======================================================================
# 7. §2 proportionality — ev_dollars scales linearly with dealer_mult
# ======================================================================
class TestEvDollarsScalesProportionallyWithDealerMultiplier:
    """CLAUDE.md §2: "the dealer multiplier ... only scales the final
    ``ev_dollars`` — it never touches ``ev_raw``". Since ``ev_raw`` is a
    local variable (``engine/ev_engine.py:366``) and not an ``EVResult``
    field, the equivalent observable invariant is **proportionality**:
    two evaluate calls with identical trade + identical forward returns,
    differing only in ``MarketStructure`` regime, produce ``ev_dollars``
    whose ratio matches the multiplier ratio.

    A regression where the multiplier accidentally touched a
    non-linear field (e.g. fed into the cost model, applied the
    multiplier to prob_profit, etc.) would break the proportionality
    and fail this test."""

    def test_ev_dollars_ratio_equals_dealer_multiplier_ratio(self):
        # Identical trade, identical forward returns, identical seed.
        # Only the MarketStructure regime differs.
        trade = _trade()
        returns = _forward_returns(seed=42)
        r_long = EVEngine().evaluate(
            trade,
            forward_log_returns=returns,
            market_structure=_market_structure("long_gamma_dampening", 1.0),
        )
        r_short = EVEngine().evaluate(
            trade,
            forward_log_returns=returns,
            market_structure=_market_structure("short_gamma_amplifying", 1.0),
        )
        # Pre-conditions for the proportionality math:
        assert r_long.dealer_multiplier == pytest.approx(1.05, abs=1e-6)
        assert r_short.dealer_multiplier == pytest.approx(0.70, abs=1e-6)
        # Sanity: trade is positive-EV (otherwise the ratio math below
        # wouldn't be meaningful at this magnitude).
        assert r_long.ev_dollars > 0
        assert r_short.ev_dollars > 0
        # The §2 invariant — expressed as observable proportionality:
        # ev_dollars ratio == dealer_multiplier ratio.
        expected_ratio = 1.05 / 0.70  # 1.5
        actual_ratio = r_long.ev_dollars / r_short.ev_dollars
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6), (
            f"ev_dollars ratio {actual_ratio:.6f} does not match expected "
            f"dealer_multiplier ratio {expected_ratio:.6f}; the dealer "
            f"multiplier is touching something beyond ev_dollars "
            f"(CLAUDE.md §2 'scales ev_dollars only' violated)"
        )
