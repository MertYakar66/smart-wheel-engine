"""
Invariant: any ChartContextProvider wired into the dossier layer flows
through a reviewer that can only downgrade, never upgrade.

Locked in to prevent future MCP-backed providers (e.g. the planned
TradingView MCP integration) from bypassing EnginePhaseReviewer's
R1-R6 rules. See CLAUDE.md §2 and §7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.candidate_dossier import build_dossiers
from engine.chart_context import ChartContext, Timeframe


@dataclass
class _FakeMCPProvider:
    """Stand-in for a hypothetical MCP-backed ChartContextProvider.

    Returns a fixed ChartContext on fetch. The test's job is to verify
    that whatever comes out of this provider *still* flows through the
    default EnginePhaseReviewer — i.e. a novel provider cannot opt out
    of rule-based review by being novel.
    """

    chart_context: ChartContext

    def fetch(self, ticker: str, timeframe: Timeframe = "1D", *, as_of=None) -> ChartContext:
        return self.chart_context


def _chart(phase: str, spot: float = 100.0) -> ChartContext:
    return ChartContext(
        ticker="FAKE",
        timeframe="1D",
        captured_at=datetime(2026, 4, 25, 12, 0, 0),
        screenshot_path=Path("/tmp/fake.png"),
        visible_price=spot,
        visible_indicators={"phase": phase},
        source="test_fake_mcp_provider",
    )


def test_custom_provider_phase_contradiction_routes_through_reviewer():
    """R4: custom provider + positive EV + contradicting phase → skip/phase_contradiction.

    Proves build_dossiers always routes through the reviewer regardless
    of which ChartContextProvider implementation is supplied.
    """
    ev_df = pd.DataFrame([{
        "ticker": "FAKE",
        "spot": 100.0,
        "strike": 95.0,
        "premium": 2.0,
        "ev_dollars": 50.0,
        "phase": "post_expansion",
    }])
    provider = _FakeMCPProvider(chart_context=_chart(phase="compression", spot=100.0))
    dossiers = build_dossiers(ev_frame=ev_df, provider=provider, top_n=5)
    assert len(dossiers) == 1
    d = dossiers[0]
    assert d.verdict == "skip", (
        f"custom provider resolved to verdict={d.verdict!r} — reviewer was not applied"
    )
    assert d.verdict_reason == "phase_contradiction"


def test_custom_provider_pristine_chart_cannot_upgrade_negative_ev():
    """R1: custom provider + pristine chart + negative EV → blocked/negative_ev.

    Downgrade-only invariant: a flawless-looking ChartContext must not
    rescue a negative-EV trade. R1 short-circuits before the chart is
    even consulted.
    """
    ev_df = pd.DataFrame([{
        "ticker": "FAKE",
        "spot": 100.0,
        "strike": 95.0,
        "premium": 2.0,
        "ev_dollars": -25.0,
        "phase": "post_expansion",
    }])
    provider = _FakeMCPProvider(chart_context=_chart(phase="post_expansion", spot=100.0))
    dossiers = build_dossiers(ev_frame=ev_df, provider=provider, top_n=5)
    assert len(dossiers) == 1
    d = dossiers[0]
    assert d.verdict == "blocked", (
        f"pristine chart rescued negative-EV trade (verdict={d.verdict!r}) — R1 breached"
    )
    assert d.verdict_reason == "negative_ev"
