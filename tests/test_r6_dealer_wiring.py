"""Audit 2026-07-15 #3: R6 dealer-positioning downgrade fires through
``build_dossiers`` via the ranker's ev_row diagnostics (``dealer_regime``,
``nearest_put_wall_strike``), not only when a ``MarketStructure`` object is
attached. Before the fix R6 was dead on the production path because
build_dossiers never threads market_structure. Downgrade-only (§2)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.candidate_dossier import build_dossiers
from engine.chart_context import ChartContext


class _CleanProvider:
    def fetch(self, ticker, timeframe="1D", *, as_of=None):
        return ChartContext(
            ticker=ticker,
            timeframe="1D",
            captured_at=datetime(2026, 4, 25, 12, 0, 0),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=100.0,
            visible_indicators={},
            source="test",
        )


def _frame(**extra):
    row = {
        "ticker": "TEST", "strike": 100.0, "premium": 2.0, "ev_dollars": 50.0,
        "iv": 0.25, "dte": 30, "spot": 100.0, "prob_profit": 0.80, "cvar_5": -5000.0,
    }
    row.update(extra)
    return pd.DataFrame([row])


def _one(ev):
    return build_dossiers(ev_frame=ev, provider=_CleanProvider(), top_n=1)[0]


def test_r6_fires_short_gamma_above_put_wall_via_ev_row():
    d = _one(_frame(dealer_regime="short_gamma_amplifying", nearest_put_wall_strike=100.0, strike=100.0))
    assert d.verdict == "review"
    assert d.verdict_reason == "dealer_short_gamma_above_put_wall"


def test_r6_fires_near_flip_via_ev_row():
    d = _one(_frame(dealer_regime="near_flip"))
    assert d.verdict == "review"
    assert d.verdict_reason == "dealer_near_flip"


def test_r6_noop_without_dealer_diagnostics():
    assert _one(_frame()).verdict == "proceed"


def test_r6_noop_when_strike_below_put_wall():
    d = _one(_frame(dealer_regime="short_gamma_amplifying", nearest_put_wall_strike=110.0, strike=100.0))
    assert d.verdict == "proceed"


def test_r6_never_rescues_blocked_negative_ev():
    # §2: R1 blocks negative EV; R6 runs only on proceed → can never upgrade.
    d = _one(_frame(ev_dollars=-50.0, dealer_regime="short_gamma_amplifying", nearest_put_wall_strike=100.0))
    assert d.verdict == "blocked"
