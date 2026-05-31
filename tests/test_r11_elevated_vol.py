"""R11 — elevated-vol top-bin size-down reviewer (heavy-verify 2026-05-31 I11).

R11 downgrades a top-bin candidate (prob_profit > 0.90) proceed→review when the
market-wide VIX *level* exceeds 25 (the I11 leave-one-crisis-out robust cut). It is
downgrade-only (never upgrades / rescues), a no-op when vix_level is absent, and its
warning payload carries the candidate's OWN computed cvar_5 (not a hardcoded number).
These pins lock that contract.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.candidate_dossier import (
    R11_TOP_BIN_PROB,
    R11_VIX_THRESHOLD,
    CandidateDossier,
    EnginePhaseReviewer,
    build_dossiers,
)
from engine.chart_context import ChartContext, Timeframe

REVIEWER = EnginePhaseReviewer()


def _dossier(prob_profit, vix_level, ev_dollars=50.0, cvar_5=-5000.0, strike=100.0):
    """A proceed-state dossier (clean chart, ev>threshold) + R11 inputs."""
    return CandidateDossier(
        ticker="TEST",
        ev_row={
            "ticker": "TEST",
            "strike": strike,
            "premium": 2.0,
            "ev_dollars": ev_dollars,
            "iv": 0.25,
            "dte": 30,
            "spot": strike,
            "prob_profit": prob_profit,
            "cvar_5": cvar_5,
        },
        chart_context=ChartContext(
            ticker="TEST",
            timeframe="1D",
            captured_at=datetime(2026, 4, 25, 12, 0, 0),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=strike,
            visible_indicators={},
            source="test",
        ),
        vix_level=vix_level,
    )


def test_r11_fires_top_bin_elevated_vol():
    v, r, notes = REVIEWER.review(_dossier(prob_profit=0.95, vix_level=30.0))
    assert v == "review"
    assert r == "elevated_vol_top_bin"
    assert any("R11" in n for n in notes)


def test_r11_noop_when_vix_below_threshold():
    v, r, _ = REVIEWER.review(_dossier(prob_profit=0.95, vix_level=20.0))
    assert v == "proceed"  # VIX not elevated → R11 silent, R5 verdict stands


def test_r11_noop_when_not_top_bin():
    v, r, _ = REVIEWER.review(_dossier(prob_profit=0.85, vix_level=30.0))
    assert v == "proceed"  # not a high-confidence candidate → R11 silent


def test_r11_noop_when_vix_absent():
    """Missing-evidence semantics (like R6-R10): no vix_level → no-op."""
    v, r, _ = REVIEWER.review(_dossier(prob_profit=0.95, vix_level=None))
    assert v == "proceed"


def test_r11_never_rescues_negative_ev():
    """R11 is gated on verdict=='proceed'; a negative-EV candidate is blocked by
    R1 long before R11 and stays blocked even with elevated VIX + top-bin prob."""
    v, r, _ = REVIEWER.review(_dossier(prob_profit=0.99, vix_level=80.0, ev_dollars=-50.0))
    assert v == "blocked"
    assert r == "negative_ev"


def test_r11_boundaries_strictly_greater_than():
    # VIX exactly at threshold → no fire; just above → fire.
    assert REVIEWER.review(_dossier(0.95, R11_VIX_THRESHOLD))[0] == "proceed"
    assert REVIEWER.review(_dossier(0.95, R11_VIX_THRESHOLD + 0.01))[0] == "review"
    # prob_profit exactly at cutoff → no fire; just above → fire.
    assert REVIEWER.review(_dossier(R11_TOP_BIN_PROB, 30.0))[0] == "proceed"
    assert REVIEWER.review(_dossier(R11_TOP_BIN_PROB + 0.001, 30.0))[0] == "review"


def test_r11_payload_is_computed_not_hardcoded():
    """The warning carries THIS candidate's own cvar_5 (regime-matched, computed),
    so it tracks the data — two candidates show two different tail figures."""
    _, _, notes_a = REVIEWER.review(_dossier(0.95, 30.0, cvar_5=-5000.0))
    _, _, notes_b = REVIEWER.review(_dossier(0.95, 30.0, cvar_5=-12345.0))
    r11_a = next(n for n in notes_a if "R11" in n)
    r11_b = next(n for n in notes_b if "R11" in n)
    assert "5,000" in r11_a and "cvar_5" in r11_a
    assert "12,345" in r11_b  # payload reflects the candidate's own modeled tail


def test_build_dossiers_threads_vix_level_and_r11_fires():
    """The wiring path: build_dossiers(vix_level=...) attaches it and R11 fires."""

    class _FakeProvider:
        def fetch(self, ticker: str, timeframe: Timeframe = "1D", *, as_of=None):
            return ChartContext(
                ticker=ticker,
                timeframe="1D",
                captured_at=datetime(2026, 4, 25, 12, 0, 0),
                screenshot_path=Path("/tmp/fake.png"),
                visible_price=100.0,
                visible_indicators={},
                source="test",
            )

    ev = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "strike": 100.0,
                "premium": 2.0,
                "ev_dollars": 50.0,
                "iv": 0.25,
                "dte": 30,
                "spot": 100.0,
                "prob_profit": 0.95,
                "cvar_5": -5000.0,
            }
        ]
    )
    armed = build_dossiers(ev_frame=ev, provider=_FakeProvider(), top_n=1, vix_level=30.0)
    assert armed[0].verdict == "review" and armed[0].verdict_reason == "elevated_vol_top_bin"
    # Same frame, no vix_level → R11 dormant, proceeds.
    dormant = build_dossiers(ev_frame=ev, provider=_FakeProvider(), top_n=1)
    assert dormant[0].verdict == "proceed"
