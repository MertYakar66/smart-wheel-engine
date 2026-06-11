"""Dealer-positioning invariants — quant-layer test audit round 2 (W60-W62), PR-5.

The dealer overlay (``engine/dealer_positioning.py``) is §2-critical: its
``dealer_regime_multiplier`` is the ONLY multiplier the engine applies to the final
``ev_dollars`` that is clamped ASYMMETRICALLY to [0.70, 1.05] (CLAUDE.md §2/§3 —
"never alter the clamp"). It scales ``ev_dollars`` only, never ``ev_raw``.

This pins (behaviour, not shape — the #366 lesson):
- W60 the [0.70, 1.05] OUTPUT clamp holds for ANY ``confidence`` (negative, >1, inf,
  nan), with the documented asymmetry (boost <= +0.05, cut down to 0.70);
- W61 the regime classification boundaries (gex==0 -> neutral; sign branches; the
  near-flip override beating the GEX sign);
- W62 the flip-distance consistency through ``analyze()``.

No §2 surface is touched — these ASSERT the clamp/contract, never weaken it.

Note: wall-ordering (put_wall <= spot <= call_wall) needs an OI-concentrated chain
that triggers wall detection (a simple chain yields None walls); deferred to a
wall-fixture follow-up. R6's at-the-put-wall boundary is a reviewer rule -> PR-7.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine.dealer_positioning import (
    DealerAssumption,
    DealerPositioningAnalyzer,
    MarketStructure,
    dealer_regime_multiplier,
)

_ASSUMPTION = list(DealerAssumption)[0]
_OUT_OF_RANGE_CONF = [-5.0, 0.0, 0.5, 1.0, 5.0, float("inf"), float("nan")]


def _ms(regime: str, confidence: float) -> MarketStructure:
    return MarketStructure(
        ticker="X",
        as_of="2026-06-04",
        spot=100.0,
        expiry="2026-07-04",
        assumption=_ASSUMPTION,
        regime=regime,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# W60 — the §2 [0.70, 1.05] clamp holds for ANY confidence. The output bound is only
# emergent from the internal confidence clamp (dealer_positioning.py:750); no test
# proves it survives an out-of-[0,1] (or non-finite) confidence. This is the §2
# invariant CLAUDE.md forbids altering.
# ---------------------------------------------------------------------------


class TestDealerClampSection2:
    @pytest.mark.parametrize(
        "regime", ["long_gamma_dampening", "short_gamma_amplifying", "near_flip", "neutral"]
    )
    @pytest.mark.parametrize("conf", _OUT_OF_RANGE_CONF)
    def test_multiplier_stays_in_clamp_for_any_confidence(self, regime, conf):
        m = dealer_regime_multiplier(_ms(regime, conf))
        assert np.isfinite(m), f"{regime} @ conf={conf}: non-finite multiplier {m}"
        assert 0.70 <= m <= 1.05, f"{regime} @ conf={conf}: {m} escaped [0.70, 1.05]"

    def test_clamp_asymmetry_and_neutral_anchor(self):
        # Asymmetric by design: long-gamma boost caps at +0.05, short-gamma cut at -0.30.
        assert dealer_regime_multiplier(_ms("long_gamma_dampening", 1.0)) == pytest.approx(1.05)
        assert dealer_regime_multiplier(_ms("short_gamma_amplifying", 1.0)) == pytest.approx(0.70)
        assert dealer_regime_multiplier(_ms("near_flip", 1.0)) == pytest.approx(0.85)
        assert dealer_regime_multiplier(_ms("neutral", 1.0)) == pytest.approx(1.0)
        # Missing structure -> neutral 1.0 (no adjustment).
        assert dealer_regime_multiplier(None) == 1.0
        # The cut is strictly larger than the boost (conservative asymmetry).
        boost = 1.05 - dealer_regime_multiplier(_ms("long_gamma_dampening", 1.0)) + 0.05
        cut = 1.0 - dealer_regime_multiplier(_ms("short_gamma_amplifying", 1.0))
        assert cut > boost, "short-gamma cut must exceed the long-gamma boost (asymmetric)"


# ---------------------------------------------------------------------------
# W61 — _classify_regime boundaries. gex_total exactly 0.0 (the long/short boundary)
# classifies 'neutral'; sign branches map to long/short; and the near-flip override
# (within flip_neighborhood_pct of the flip) beats the GEX sign. Only strictly >0 / <0
# were tested; the 0.0 boundary + the override were not.
# ---------------------------------------------------------------------------


class TestClassifyRegimeBoundaries:
    def test_gex_sign_and_zero_boundary(self):
        az = DealerPositioningAnalyzer()
        assert az._classify_regime(0.0, None) == "neutral"
        assert az._classify_regime(5.0, None) == "long_gamma_dampening"
        assert az._classify_regime(-5.0, None) == "short_gamma_amplifying"

    def test_near_flip_overrides_gex_sign(self):
        az = DealerPositioningAnalyzer()
        tiny = az.flip_neighborhood_pct / 2.0
        # Strong positive GEX but within the flip band -> near_flip wins.
        assert az._classify_regime(5.0, tiny) == "near_flip"
        assert az._classify_regime(-5.0, -tiny) == "near_flip"
        # Just outside the band -> the GEX sign governs again.
        assert az._classify_regime(5.0, az.flip_neighborhood_pct * 2) == "long_gamma_dampening"


# ---------------------------------------------------------------------------
# W62 — flip-distance consistency through analyze(). When a flip level is found,
# flip_distance_pct must equal (flip_level - spot) / spot (dealer_positioning.py:383).
# Never pinned end-to-end.
# ---------------------------------------------------------------------------


class TestFlipDistanceConsistency:
    def test_flip_distance_pct_matches_flip_level(self):
        rows = []
        for k in range(80, 121, 5):
            rows.append(
                {
                    "strike": float(k),
                    "option_type": "call",
                    "open_interest": 2000 if k >= 110 else 500,
                    "implied_vol": 0.25,
                }
            )
            rows.append(
                {
                    "strike": float(k),
                    "option_type": "put",
                    "open_interest": 2000 if k <= 90 else 500,
                    "implied_vol": 0.25,
                }
            )
        chain = pd.DataFrame(rows)
        ms = DealerPositioningAnalyzer().analyze(
            chain, spot=100.0, expiry=date(2026, 7, 4), ticker="X"
        )
        assert ms.flip_level is not None and ms.flip_distance_pct is not None, (
            "expected a flip level on this chain"
        )
        assert ms.flip_distance_pct == pytest.approx((ms.flip_level - 100.0) / 100.0)
        # Sanity: a found flip on a near-ATM chain stays within a sane band.
        assert -0.5 < ms.flip_distance_pct < 0.5
