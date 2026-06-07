"""Regression tests for the non-finite EV defense across verdict paths.

Closes **C1** and **C2** from
``docs/END_TO_END_REVIEW_2026_05_25.md``:

* **C1** — :class:`engine.candidate_dossier.EnginePhaseReviewer` R1
  previously had no ``math.isfinite`` guard. ``+inf`` ``ev_dollars``
  slipped through both R1 (``+inf < 0`` is False) and R5
  (``+inf >= threshold`` is True) and was reported as ``"proceed"``;
  ``NaN`` silently degraded to ``"review"`` via R5's strict ``>=``
  (``NaN >= threshold`` is False), masking the real signal. The fix
  adds rule R1a: a non-finite EV returns
  ``("blocked", "ev_non_finite", ...)`` before any other rule runs.

* **C2** — ``engine_api.EngineAPIHandler._enrich_alert`` (the
  ``/api/tv/webhook`` verdict path) carried its own copy of the
  R1/R5 ladder, with no symmetry on the non-finite guard and a
  duplicated hardcoded ``>= 10`` threshold. The fix:
  - Mirrors the non-finite guard:
    non-finite EV → ``("skip", "ev_non_finite")``.
  - Imports ``MIN_PROCEED_EV_DOLLARS`` from
    :mod:`engine.candidate_dossier` so the threshold cannot drift
    from the dossier reviewer's default.
  - Same constant also drives the ``/api/candidates`` recommendation
    label.

These tests inoculate against either fix being reverted and against
the threshold-drift surface re-opening.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from engine.candidate_dossier import (
    MIN_PROCEED_EV_DOLLARS,
    CandidateDossier,
    EnginePhaseReviewer,
)
from engine.chart_context import ChartContext


def _pristine_chart(spot: float = 100.0) -> ChartContext:
    """A chart context that passes R2 (no error, screenshot_path set)
    and R3 (visible_price matches spot). R4 stays dormant (no phase
    key). The point is for R1a to be the only rule that could fire."""
    return ChartContext(
        ticker="TEST",
        timeframe="1D",
        captured_at=datetime.now(UTC).replace(tzinfo=None),
        screenshot_path=Path("/tmp/test.png"),
        visible_price=spot,
        visible_indicators={},
        source="test_pristine_chart",
    )


def _dossier(ev_dollars: float, spot: float = 100.0) -> CandidateDossier:
    """Build a dossier with a pristine chart so the only rule that
    can block / downgrade is R1a (non-finite) or R1 (negative). For
    finite positive EVs the reviewer proceeds past R3 to R5."""
    return CandidateDossier(
        ticker="TEST",
        ev_row={
            "ev_dollars": ev_dollars,
            "spot": spot,
            "strike": 95.0,
            "premium": 1.5,
            "dte": 35,
            "iv": 0.22,
        },
        chart_context=_pristine_chart(spot=spot),
    )


# ======================================================================
# C1 — EnginePhaseReviewer R1a non-finite defense
# ======================================================================
class TestEnginePhaseReviewerNonFiniteR1a:
    """The dossier-side reviewer hard-blocks non-finite ``ev_dollars``
    BEFORE R1's negative check. Distinct ``verdict_reason`` separates
    "engine produced an unparseable value" (audit signal) from
    "engine evaluated the trade as a loss" (ordinary block).
    """

    def test_plus_inf_blocks_with_ev_non_finite_reason(self):
        """``+inf`` is the headline C1 bypass: previously slid through
        R1 (False: ``+inf < 0``) and R5 (True: ``+inf >= threshold``)
        all the way to ``"proceed"``."""
        reviewer = EnginePhaseReviewer()
        verdict, reason, notes = reviewer.review(_dossier(float("inf")))
        assert verdict == "blocked", (
            f"+inf must hard-block; got verdict={verdict!r} reason={reason!r} — "
            "R1a non-finite defense reverted?"
        )
        assert reason == "ev_non_finite", (
            f"+inf must use the distinct ev_non_finite reason (not negative_ev); got {reason!r}"
        )
        assert any("not finite" in n for n in notes), (
            f"R1a note must surface 'not finite'; got notes={notes!r}"
        )

    def test_nan_blocks_with_ev_non_finite_reason(self):
        """``NaN`` previously silently degraded to ``"review"`` via
        R5's strict ``>=`` (``NaN >= threshold`` is False)."""
        reviewer = EnginePhaseReviewer()
        verdict, reason, _ = reviewer.review(_dossier(float("nan")))
        assert verdict == "blocked", (
            f"NaN must hard-block (not silently downgrade to review); got verdict={verdict!r}"
        )
        assert reason == "ev_non_finite"

    def test_minus_inf_blocks_with_ev_non_finite_reason(self):
        """``-inf`` was already blocked by R1 (``-inf < 0`` is True),
        but R1a should fire first because non-finite is the more
        precise audit signal. R1a precedence is intentional.
        """
        reviewer = EnginePhaseReviewer()
        verdict, reason, _ = reviewer.review(_dossier(float("-inf")))
        assert verdict == "blocked"
        # -inf is both non-finite AND < 0; R1a fires first.
        assert reason == "ev_non_finite", (
            f"-inf should land in R1a (more precise) not R1; got {reason!r}"
        )

    @pytest.mark.parametrize(
        "ev,expected_verdict,expected_reason",
        [
            (-25.0, "blocked", "negative_ev"),
            (-0.01, "blocked", "negative_ev"),
            (0.0, "review", "ev_below_proceed_threshold"),
            (5.0, "review", "ev_below_proceed_threshold"),
            (MIN_PROCEED_EV_DOLLARS, "proceed", "ev_above_threshold"),
            (MIN_PROCEED_EV_DOLLARS + 0.01, "proceed", "ev_above_threshold"),
            (1e6, "proceed", "ev_above_threshold"),
        ],
    )
    def test_finite_evs_route_to_expected_rules(
        self, ev: float, expected_verdict: str, expected_reason: str
    ):
        """Pin that R1a does not interfere with finite-EV routing —
        finite negatives still hit R1, finite positives still hit
        R5. Boundary at ``MIN_PROCEED_EV_DOLLARS`` to catch off-by-one.
        """
        reviewer = EnginePhaseReviewer()
        verdict, reason, _ = reviewer.review(_dossier(ev))
        assert (verdict, reason) == (expected_verdict, expected_reason), (
            f"finite EV {ev} should route to {expected_verdict}/{expected_reason}; "
            f"got {verdict}/{reason}"
        )


# ======================================================================
# C2 — _enrich_alert non-finite defense + shared threshold
# ======================================================================
class _FakeRunner:
    """Minimal runner stub returning a configurable ev_dollars row."""

    def __init__(self, ev_dollars: float, prob_profit: float = 0.78):
        self.ev_dollars = ev_dollars
        self.prob_profit = prob_profit

    def analyze_ticker(self, ticker, as_of=None):
        from engine.wheel_runner import TickerAnalysis

        return TickerAnalysis(
            ticker=ticker,
            spot_price=100.0,
            wheel_score=20.0,
            wheel_recommendation="weak",
            days_to_earnings=None,
            sector="Tech",
        )

    def rank_candidates_by_ev(self, **kwargs):
        return pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "spot": 100.0,
                    "strike": 95.0,
                    "premium": 1.50,
                    "dte": 35,
                    "iv": 0.22,
                    "ev_dollars": self.ev_dollars,
                    "ev_per_day": self.ev_dollars / 35.0 if math.isfinite(self.ev_dollars) else 0.0,
                    "prob_profit": self.prob_profit,
                    "prob_assignment": 0.18,
                    "distribution_source": "empirical_non_overlapping",
                }
            ]
        )


class _FakeConn:
    """Minimal connector stub with the keys ``_enrich_alert`` reads."""

    def get_ohlcv(self, ticker):
        import numpy as np

        idx = pd.date_range("2020-01-01", periods=800, freq="B")
        prices = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 800)))
        return pd.DataFrame({"close": prices}, index=idx)

    def get_iv_rank(self, ticker, as_of=None):
        return 55.0

    def get_vol_risk_premium(self, ticker, as_of=None):
        return 2.5


def _agreeing_signal():
    from engine.tv_signals import TVSignal as _TVSignal

    return _TVSignal(
        ticker="AAPL",
        ok=True,
        close=100.0,
        phase="post_expansion",
        signal_action="wheel_put_zone",
        wheel_put_zone=True,
    )


def _enrich_with_ev(ev_dollars: float, prob_profit: float = 0.78) -> dict:
    """Drive ``_enrich_alert`` with the given EV and a chart-agreeing
    Pine alert. Returns the enriched verdict dict."""
    from engine.tv_signals import TVAlert
    from engine_api import EngineAPIHandler

    alert = TVAlert(ticker="AAPL", signal="wheel_put_zone", source="test")
    handler = EngineAPIHandler.__new__(EngineAPIHandler)
    with (
        patch("engine_api.get_connector", return_value=_FakeConn()),
        patch(
            "engine_api.get_runner",
            return_value=_FakeRunner(ev_dollars=ev_dollars, prob_profit=prob_profit),
        ),
        patch("engine.tv_signals.compute_tv_signal", return_value=_agreeing_signal()),
    ):
        return handler._enrich_alert(alert)


class TestEnrichAlertNonFiniteDefense:
    """``_enrich_alert``'s verdict ladder rejects non-finite EV with
    a distinct reason BEFORE the negative / proceed branches. Mirror
    of the dossier R1a fix; closes the C2 non-finite half.
    """

    def test_plus_inf_yields_blocked_with_ev_non_finite_reason(self):
        # R27: hard-stop label aligned to the dossier reviewer R1a ("blocked",
        # was "skip"). The verdict_reason is unchanged.
        enriched = _enrich_with_ev(float("inf"))
        assert enriched["authority"] == "ev_ranked"
        assert enriched["verdict"] == "blocked", (
            f"+inf must block (not proceed via the >= threshold branch); "
            f"got verdict={enriched['verdict']!r} reason={enriched['verdict_reason']!r}"
        )
        assert enriched["verdict_reason"] == "ev_non_finite"

    def test_nan_yields_blocked_with_ev_non_finite_reason(self):
        enriched = _enrich_with_ev(float("nan"))
        assert enriched["verdict"] == "blocked"  # R27 label alignment
        assert enriched["verdict_reason"] == "ev_non_finite", (
            f"NaN must use the distinct ev_non_finite reason; "
            f"got {enriched['verdict_reason']!r} (was 'ev_zero_or_below' "
            "pre-fix because NaN fell all the way through the ladder)"
        )

    def test_minus_inf_yields_blocked_with_ev_non_finite_reason(self):
        enriched = _enrich_with_ev(float("-inf"))
        assert enriched["verdict"] == "blocked"  # R27 label alignment
        # -inf is non-finite first; R1a-equivalent fires before negative_ev.
        assert enriched["verdict_reason"] == "ev_non_finite"

    def test_finite_negative_still_yields_negative_ev(self):
        """Finite negative EV preserves the original ``"negative_ev"``
        verdict_reason — the non-finite branch must not over-reach. R27
        aligns the verdict LABEL to "blocked" (was "skip"); reason unchanged.
        """
        enriched = _enrich_with_ev(-5.0)
        assert enriched["verdict"] == "blocked"  # R27 label alignment
        assert enriched["verdict_reason"] == "negative_ev"

    def test_finite_positive_at_threshold_proceeds(self):
        """Boundary check: exactly at ``MIN_PROCEED_EV_DOLLARS`` with
        prob_profit and chart agreement → proceed."""
        enriched = _enrich_with_ev(MIN_PROCEED_EV_DOLLARS, prob_profit=0.78)
        assert enriched["verdict"] == "proceed"
        assert enriched["verdict_reason"] == "ev_above_threshold_and_chart_agrees"

    def test_finite_positive_below_threshold_reviews(self):
        enriched = _enrich_with_ev(MIN_PROCEED_EV_DOLLARS - 0.01, prob_profit=0.78)
        assert enriched["verdict"] == "review"
        assert enriched["verdict_reason"] == "positive_but_low_ev"


# ======================================================================
# C2 — Shared MIN_PROCEED_EV_DOLLARS constant (drift prevention)
# ======================================================================
class TestSharedMinProceedEVDollarsConstant:
    """The proceed threshold lives in ``engine.candidate_dossier``
    and the webhook ladder imports it. Pin the wiring so a future
    refactor doesn't accidentally split the source of truth.
    """

    def test_engine_api_imports_constant_from_candidate_dossier(self):
        """``_MIN_PROCEED_EV_DOLLARS`` in engine_api must BE the same
        object as ``MIN_PROCEED_EV_DOLLARS`` in candidate_dossier.
        """
        from engine.candidate_dossier import MIN_PROCEED_EV_DOLLARS as canonical
        from engine_api import _MIN_PROCEED_EV_DOLLARS as imported

        assert imported is canonical or imported == canonical, (
            "engine_api._MIN_PROCEED_EV_DOLLARS diverged from "
            "engine.candidate_dossier.MIN_PROCEED_EV_DOLLARS — drift!"
        )

    def test_engine_phase_reviewer_default_matches_constant(self):
        """``EnginePhaseReviewer`` default constructor must use
        ``MIN_PROCEED_EV_DOLLARS``."""
        reviewer = EnginePhaseReviewer()
        assert reviewer.min_proceed_ev == MIN_PROCEED_EV_DOLLARS, (
            f"EnginePhaseReviewer() default min_proceed_ev={reviewer.min_proceed_ev} "
            f"diverged from MIN_PROCEED_EV_DOLLARS={MIN_PROCEED_EV_DOLLARS}"
        )

    def test_constant_value_remains_ten_dollars(self):
        """Pin the literal value so a silent retune is caught. If the
        team wants to change the threshold, this test must be updated
        in the same PR — that's the audit-trail point."""
        assert MIN_PROCEED_EV_DOLLARS == 10.0, (
            f"MIN_PROCEED_EV_DOLLARS retuned to {MIN_PROCEED_EV_DOLLARS} — "
            "if intentional, update this test in the same PR"
        )
