"""Regression: EnginePhaseReviewer review-notes must be cp1252-safe.

S1 logged a U+0394 (Greek capital delta) character in the R3
spot-mismatch note; printing or logging it on a Windows cp1252 console
raised UnicodeEncodeError. The P2 fix (issue #118) makes every reviewer
note string ASCII. This exercises each note-producing rule path
(R1, chart-missing, R3, R4, R5, R6) and asserts every emitted note
survives ``str.encode("cp1252")``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer
from engine.chart_context import ChartContext

_DELTA = chr(0x394)  # U+0394 Greek capital delta; S1's R3 note carried it


def _chart(visible_price: float = 100.0, phase: str = "post_expansion") -> ChartContext:
    """A well-formed (is_ok) ChartContext."""
    return ChartContext(
        ticker="FAKE",
        timeframe="1D",
        captured_at=datetime(2026, 4, 25, 12, 0, 0),
        screenshot_path=Path("/tmp/fake.png"),
        visible_price=visible_price,
        visible_indicators={"phase": phase},
        source="test_dossier_cp1252",
    )


def _dossier(ev_row: dict, chart=None, market_structure=None) -> CandidateDossier:
    return CandidateDossier(
        ticker="FAKE",
        ev_row=ev_row,
        chart_context=chart,
        market_structure=market_structure,
    )


def _review_note_cases() -> list[tuple[str, CandidateDossier]]:
    """One dossier per note-producing branch of EnginePhaseReviewer.review()."""
    base = {"spot": 100.0, "strike": 100.0, "premium": 2.0, "phase": "post_expansion"}
    short_gamma = SimpleNamespace(
        regime="short_gamma_amplifying",
        nearest_put_wall=SimpleNamespace(strike=95.0),
    )
    near_flip = SimpleNamespace(regime="near_flip", nearest_put_wall=None)
    return [
        # R1 - negative EV is blocked
        ("R1_negative_ev", _dossier({**base, "ev_dollars": -25.0})),
        # chart missing -> review
        ("chart_missing", _dossier({**base, "ev_dollars": 50.0}, chart=None)),
        # R3 - spot mismatch -> skip (the note that carried the delta char)
        (
            "R3_spot_mismatch",
            _dossier({**base, "ev_dollars": 50.0}, chart=_chart(visible_price=120.0)),
        ),
        # R3 agree + R5 proceed
        (
            "R3_agree_R5_proceed",
            _dossier({**base, "ev_dollars": 50.0}, chart=_chart(visible_price=100.0)),
        ),
        # R4 - phase contradiction -> skip
        (
            "R4_phase_contradiction",
            _dossier({**base, "ev_dollars": 50.0}, chart=_chart(phase="compression")),
        ),
        # R5 - EV below proceed threshold -> review
        (
            "R5_review",
            _dossier({**base, "ev_dollars": 5.0}, chart=_chart(visible_price=100.0)),
        ),
        # R6 - short-gamma regime + strike at/above put wall -> review
        (
            "R6_short_gamma",
            _dossier(
                {**base, "ev_dollars": 50.0},
                chart=_chart(visible_price=100.0),
                market_structure=short_gamma,
            ),
        ),
        # R6 - dealer regime near gamma flip -> review
        (
            "R6_near_flip",
            _dossier(
                {**base, "ev_dollars": 50.0},
                chart=_chart(visible_price=100.0),
                market_structure=near_flip,
            ),
        ),
    ]


def test_all_reviewer_notes_encode_to_cp1252():
    """Every note from every reviewer rule path encodes to cp1252 without
    raising. A non-cp1252 char crashes a Windows console on print/log."""
    reviewer = EnginePhaseReviewer()
    for label, dossier in _review_note_cases():
        _verdict, _reason, notes = reviewer.review(dossier)
        assert notes, f"{label}: review() produced no notes"
        for note in notes:
            try:
                note.encode("cp1252")
            except UnicodeEncodeError as exc:
                raise AssertionError(f"{label}: review note is not cp1252-safe: {note!r}") from exc


def test_r3_spot_mismatch_note_present_and_delta_free():
    """The R3 spot-mismatch note (which carried U+0394 in S1) is emitted,
    cp1252-safe, and no longer contains the Greek delta character."""
    reviewer = EnginePhaseReviewer()
    dossier = _dossier(
        {
            "spot": 100.0,
            "strike": 100.0,
            "premium": 2.0,
            "phase": "post_expansion",
            "ev_dollars": 50.0,
        },
        chart=_chart(visible_price=120.0),
    )
    verdict, reason, notes = reviewer.review(dossier)
    assert verdict == "skip"
    assert reason == "spot_price_mismatch"
    r3 = next((n for n in notes if "disagrees with engine spot" in n), None)
    assert r3 is not None, f"R3 spot-mismatch note not found in {notes}"
    assert _DELTA not in r3, "R3 note still contains the U+0394 delta character"
    r3.encode("cp1252")  # must not raise
