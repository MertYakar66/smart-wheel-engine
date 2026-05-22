"""Tests for the strangle-recommendation phase/confidence gate.

Issue #118 P5. S14 logged that `StrangleEntryScore.recommendation` was
a pure `total_score` cut — it ignored the `VolatilityPhase` and the
confidence `_classify_phase` already computes. Two concrete defects:
MSFT scored 81.6 → `strong_entry` while its phase was `unknown`
(confidence 0.30); and `VolatilityPhase.TREND` is documented "AVOID"
but had no recommendation override.

The fix adds `_apply_phase_gate` — a **strictly downgrade-only** gate —
to both `score_entry` (Layer 1) and `StrangleTimingWithIV.score_entry`
(the IV path, where the MSFT score came from), and surfaces
`phase_confidence` on `StrangleEntryScore`.

Pinned here:
  * the gate is downgrade-only — a good phase / high confidence never
    lifts a recommendation, and an unchanged base passes through;
  * `VolatilityPhase.TREND` → `avoid`;
  * an `unknown` phase or sub-floor confidence caps `strong_entry` at
    `conditional` (the MSFT case) — Layer 1 and the IV path both;
  * `phase_confidence` is surfaced on `StrangleEntryScore`.
"""

from __future__ import annotations

import pandas as pd

from engine.strangle_timing import (
    StrangleEntryScore,
    StrangleRegime,
    StrangleTimingEngine,
    StrangleTimingWithIV,
    VolatilityPhase,
)

_FLOOR = StrangleTimingEngine._PHASE_CONFIDENCE_FLOOR  # 0.70


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _regime(
    phase: VolatilityPhase,
    confidence: float,
    *,
    strong: bool = True,
) -> StrangleRegime:
    """A StrangleRegime with the given phase/confidence. ``strong`` picks
    component states that score high (`score_entry` total ~89) or low."""
    if strong:
        return StrangleRegime(
            phase=phase,
            confidence=confidence,
            bollinger_state="wide_contracting",
            atr_state="declining",
            rsi_state="extreme_ob",
            trend_state="flat",
            range_state="beyond_band",
        )
    return StrangleRegime(
        phase=phase,
        confidence=confidence,
        bollinger_state="narrow",
        atr_state="low",
        rsi_state="neutral",
        trend_state="strong_up",
        range_state="mid",
    )


def _engine_with_regime(regime: StrangleRegime) -> StrangleTimingEngine:
    """A StrangleTimingEngine whose classify_regime always returns ``regime``
    — so score_entry is exercised against a controlled phase/confidence."""
    eng = StrangleTimingEngine()
    eng.classify_regime = lambda df: regime  # type: ignore[method-assign]
    return eng


# ======================================================================
# 1. _apply_phase_gate — the gate, in isolation
# ======================================================================
class TestApplyPhaseGate:
    def setup_method(self):
        self.eng = StrangleTimingEngine()

    def test_trend_phase_forces_avoid(self):
        g = self.eng._apply_phase_gate
        trend = _regime(VolatilityPhase.TREND, 0.75)
        assert g("strong_entry", trend) == "avoid"
        assert g("conditional", trend) == "avoid"
        assert g("avoid", trend) == "avoid"

    def test_unknown_phase_caps_strong_entry_at_conditional(self):
        g = self.eng._apply_phase_gate
        unknown = _regime(VolatilityPhase.UNKNOWN, 0.30)
        assert g("strong_entry", unknown) == "conditional"  # the MSFT case
        assert g("conditional", unknown) == "conditional"
        assert g("avoid", unknown) == "avoid"

    def test_low_confidence_caps_strong_entry(self):
        """A weak read of an otherwise-good phase (POST_EXPANSION @ 0.60)
        still cannot reach strong_entry."""
        g = self.eng._apply_phase_gate
        weak = _regime(VolatilityPhase.POST_EXPANSION, 0.60)
        assert g("strong_entry", weak) == "conditional"

    def test_confident_good_phase_leaves_strong_entry_intact(self):
        """A confident POST_EXPANSION (0.90 ≥ floor) does not get capped —
        the gate only ever downgrades, it must not touch a legitimate
        strong_entry."""
        g = self.eng._apply_phase_gate
        good = _regime(VolatilityPhase.POST_EXPANSION, 0.90)
        assert g("strong_entry", good) == "strong_entry"

    def test_gate_never_upgrades(self):
        """Downgrade-only: a good phase / high confidence must never lift
        a recommendation above its score-cut base."""
        g = self.eng._apply_phase_gate
        good = _regime(VolatilityPhase.POST_EXPANSION, 0.90)
        assert g("conditional", good) == "conditional"
        assert g("avoid", good) == "avoid"
        # even EXPANSION/COMPRESSION phases passed here never upgrade
        assert g("avoid", _regime(VolatilityPhase.EXPANSION, 0.80)) == "avoid"

    def test_confidence_floor_boundary(self):
        """`< floor` is strict: confidence exactly at the floor is not low."""
        g = self.eng._apply_phase_gate
        at_floor = _regime(VolatilityPhase.POST_EXPANSION, _FLOOR)
        below = _regime(VolatilityPhase.POST_EXPANSION, _FLOOR - 0.01)
        assert g("strong_entry", at_floor) == "strong_entry"
        assert g("strong_entry", below) == "conditional"

    def test_none_regime_passes_through(self):
        assert self.eng._apply_phase_gate("strong_entry", None) == "strong_entry"


# ======================================================================
# 2. phase_confidence surfaced on StrangleEntryScore
# ======================================================================
class TestPhaseConfidenceField:
    def test_field_exists_with_default(self):
        score = StrangleEntryScore(total_score=70.0, recommendation="conditional")
        assert score.phase_confidence == 0.0

    def test_score_entry_populates_phase_confidence(self):
        regime = _regime(VolatilityPhase.POST_EXPANSION, 0.90)
        score = _engine_with_regime(regime).score_entry(pd.DataFrame())
        assert score.phase_confidence == 0.90
        assert score.phase_confidence == regime.confidence


# ======================================================================
# 3. score_entry — Layer 1 integration
# ======================================================================
class TestScoreEntryGate:
    def test_trend_phase_score_entry_returns_avoid(self):
        regime = _regime(VolatilityPhase.TREND, 0.75)
        score = _engine_with_regime(regime).score_entry(pd.DataFrame())
        assert score.recommendation == "avoid"

    def test_unknown_phase_high_score_capped_at_conditional(self):
        """The MSFT case: a top-tier total_score on an unknown-phase
        regime must not return strong_entry."""
        regime = _regime(VolatilityPhase.UNKNOWN, 0.30, strong=True)
        score = _engine_with_regime(regime).score_entry(pd.DataFrame())
        assert score.total_score >= 80.0  # base cut would say strong_entry
        assert score.recommendation == "conditional"  # ...but the gate caps it

    def test_confident_good_phase_high_score_stays_strong_entry(self):
        """A confident POST_EXPANSION with a top-tier score is exactly
        what strong_entry is for — the gate must leave it alone."""
        regime = _regime(VolatilityPhase.POST_EXPANSION, 0.90, strong=True)
        score = _engine_with_regime(regime).score_entry(pd.DataFrame())
        assert score.total_score >= 80.0
        assert score.recommendation == "strong_entry"


# ======================================================================
# 4. StrangleTimingWithIV.score_entry — the IV path also gates
# ======================================================================
class TestIVPathGate:
    _NEUTRAL_IV = {
        "iv_rank": 50.0,
        "vol_risk_premium": 0.0,
        "vix_level": 20.0,
        "vix_contango": True,
    }

    def _iv_engine(self, regime: StrangleRegime) -> StrangleTimingWithIV:
        eng = StrangleTimingWithIV()
        eng.classify_regime = lambda df: regime  # type: ignore[method-assign]
        return eng

    def test_iv_path_caps_unknown_phase_strong_entry(self):
        """S14's MSFT score came through the IV overlay — the IV path
        re-derives its own recommendation, so it must gate too."""
        regime = _regime(VolatilityPhase.UNKNOWN, 0.30, strong=True)
        score = self._iv_engine(regime).score_entry(pd.DataFrame(), iv_data=self._NEUTRAL_IV)
        assert score.total_score >= 80.0
        assert score.recommendation == "conditional"

    def test_iv_path_trend_phase_returns_avoid(self):
        regime = _regime(VolatilityPhase.TREND, 0.75, strong=True)
        score = self._iv_engine(regime).score_entry(pd.DataFrame(), iv_data=self._NEUTRAL_IV)
        assert score.recommendation == "avoid"

    def test_iv_path_surfaces_phase_confidence(self):
        regime = _regime(VolatilityPhase.POST_EXPANSION, 0.90, strong=True)
        score = self._iv_engine(regime).score_entry(pd.DataFrame(), iv_data=self._NEUTRAL_IV)
        assert score.phase_confidence == 0.90

    def test_iv_path_none_iv_data_is_layer1_gated(self):
        """With iv_data=None the IV path returns the Layer-1 score
        verbatim — already gated."""
        regime = _regime(VolatilityPhase.TREND, 0.75)
        score = self._iv_engine(regime).score_entry(pd.DataFrame(), iv_data=None)
        assert score.recommendation == "avoid"


# ======================================================================
# 5. §2 / no-regression — rank_strangles_by_ev still EV-ranks
# ======================================================================
class TestRankStranglesUnaffected:
    def test_rank_strangles_by_ev_still_produces_candidates(self):
        """rank_strangles_by_ev consumes strangle_timing as a
        downgrade-only pre-filter; producing more avoids cannot change
        the EV ranking itself. The EV path is unchanged."""
        from engine.wheel_runner import WheelRunner

        runner = WheelRunner()
        df = runner.rank_strangles_by_ev(
            "AAPL", as_of="2026-03-20", min_ev_dollars=-1e9, use_timing_gate=False
        )
        assert len(df) == 16
        assert "ev_dollars" in df.columns
