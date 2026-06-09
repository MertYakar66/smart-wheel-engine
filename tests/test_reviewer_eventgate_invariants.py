"""Decision-reviewer + event-gate boundary invariants — quant-layer test audit
round 2 (W65-W67), PR-7.

Pins the EXACT boundaries of the §2 downgrade-only reviewer rules
(``engine/candidate_dossier.py`` EnginePhaseReviewer) and the event lockout
(``engine/event_gate.py``):

- W65 R5's inclusive ``ev_dollars >= min_proceed_ev`` boundary (== threshold ->
  proceed; a stealth flip to strict ``>`` would silently demote at-threshold
  candidates to review).
- W66 R3's spot-mismatch: strict ``diff > tol`` (== tol does NOT skip) and the
  ``engine_spot > 0`` guard that skips the check on a degenerate spot.
- W67 EventGate.is_blocked returns the EARLIEST in-window event regardless of
  wildcard-macro vs ticker-specific (only the same-ticker multi-earnings case was
  tested).

These ASSERT the §2 contract (reviewers downgrade proceed->review/skip/blocked,
never upgrade); no §2 surface is weakened. EnginePhaseReviewer is constructed with
an explicit ``min_proceed_ev`` so the boundary is deterministic.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer
from engine.chart_context import ChartContext
from engine.event_gate import EventGate, ScheduledEvent

_MIN_PROCEED = 10.0


def _chart(visible_price: float) -> ChartContext:
    return ChartContext(
        ticker="X",
        timeframe="1D",
        captured_at=datetime(2026, 6, 4, 12, 0, 0),
        screenshot_path=Path("/tmp/x.png"),
        visible_price=visible_price,
        visible_indicators={},
        source="probe",
    )


def _review(ev: float, spot: float, chart_price: float):
    dossier = CandidateDossier(
        ticker="X",
        ev_row={"ev_dollars": ev, "spot": spot},
        chart_context=_chart(chart_price),
    )
    verdict, reason, _notes = EnginePhaseReviewer(min_proceed_ev=_MIN_PROCEED).review(dossier)
    return verdict, reason


# ---------------------------------------------------------------------------
# W65 — R5 inclusive threshold boundary.
# ---------------------------------------------------------------------------


class TestR5ThresholdBoundary:
    def test_ev_exactly_at_threshold_proceeds(self):
        # ev == min_proceed_ev -> proceed (inclusive >=). Spot agrees so R3 is a no-op.
        verdict, reason = _review(_MIN_PROCEED, 100.0, 100.0)
        assert verdict == "proceed" and reason == "ev_above_threshold"

    def test_ev_just_below_threshold_reviews(self):
        verdict, reason = _review(_MIN_PROCEED - 0.01, 100.0, 100.0)
        assert verdict == "review" and reason == "ev_below_proceed_threshold"


# ---------------------------------------------------------------------------
# W66 — R3 spot-mismatch strictness + the engine_spot>0 guard.
# ---------------------------------------------------------------------------


class TestR3SpotMismatchBoundary:
    def test_diff_above_tolerance_skips(self):
        # spot 100, chart 105 => 5% > 2% default tol -> skip (a downgrade from proceed).
        verdict, reason = _review(50.0, 100.0, 105.0)
        assert verdict == "skip" and reason == "spot_price_mismatch"

    def test_diff_exactly_at_tolerance_does_not_skip(self):
        # diff == tol (2.00 == 100*0.02): strict `diff > tol` -> NOT skipped -> proceeds.
        verdict, reason = _review(50.0, 100.0, 102.0)
        assert verdict == "proceed", f"diff==tol must not skip (strict >), got {verdict}/{reason}"

    def test_nonpositive_engine_spot_skips_the_r3_check(self):
        # engine_spot <= 0 -> the R3 guard skips the spot check even on a gross
        # disagreement, so the candidate continues to R5 (no spot_price_mismatch).
        verdict, reason = _review(50.0, 0.0, 105.0)
        assert reason != "spot_price_mismatch"
        assert verdict == "proceed"


# ---------------------------------------------------------------------------
# W67 — EventGate earliest-hit across mixed wildcard/specific events.
# ---------------------------------------------------------------------------


class TestEventGateEarliestHit:
    def test_earliest_in_window_event_wins_regardless_of_wildcard(self):
        gate = EventGate(
            events=[
                ScheduledEvent(ticker="JPM", kind="earnings", event_date=date(2026, 6, 20)),
                ScheduledEvent(
                    ticker="*", kind="fomc", event_date=date(2026, 6, 12)
                ),  # earlier, wildcard
            ]
        )
        blocked, reason = gate.is_blocked("JPM", date(2026, 6, 10), date(2026, 6, 25))
        assert blocked is True
        # The earliest event (the wildcard FOMC) is the reported reason, not the
        # later ticker-specific earnings.
        assert "fomc" in reason and "2026-06-12" in reason, reason

    def test_no_events_in_window_is_not_blocked(self):
        gate = EventGate(
            events=[ScheduledEvent(ticker="JPM", kind="earnings", event_date=date(2026, 9, 1))]
        )
        blocked, reason = gate.is_blocked("JPM", date(2026, 6, 10), date(2026, 6, 25))
        assert blocked is False and reason == ""
